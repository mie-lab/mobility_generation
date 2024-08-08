import copy
import functools
import os
import time
import glob

import blobfile as bf
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import math

torch.autograd.set_detect_anomaly(True)

import pickle as pickle

from utils import logger
from utils.dist_util import get_device, load_state_dict
from utils.min_norm_solvers import MinNormSolver, gradient_normalizers

from generative.diff.step_sample import LossAwareSampler


def _get_sqrt_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, min_decay: float, num_training_steps: int, num_cycles: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return min_decay
    return max(min_decay, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


def get_sqrt_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_decay: float,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):

    lr_lambda = partial(
        _get_sqrt_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_decay=min_decay,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diff_steps,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        warmup_epochs=2,
        decay_epochs=100,
        max_epochs=150,
        save_epochs=5,
        use_fp16=False,
        schedule_sampler=None,
        weight_decay=0.0,
        checkpoint_path="",
        load_checkpoint=False,
        load_opt=True,
        gradient_clipping=-1.0,
        eval_data=None,
        self_cond=False,
    ):
        self.model = model
        self.diff_steps = diff_steps
        self.data = data
        self.eval_data = eval_data

        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.grad_accum_steps = self.batch_size / (self.microbatch)

        self.lr = lr
        self.ema_rate = ema_rate
        self.log_interval = log_interval
        self.decay_epochs = decay_epochs
        self.use_fp16 = use_fp16
        self.schedule_sampler = schedule_sampler
        self.gradient_clipping = gradient_clipping

        self.step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.max_epochs = max_epochs
        self.save_epochs = save_epochs

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

        self.checkpoint_path = checkpoint_path

        # control learning rate
        self.opt = self.model.configure_optimizers(
            weight_decay=weight_decay, learning_rate=self.lr, betas=(0.9, 0.98), device_type="cuda"
        )
        # define learning rate schedule
        if load_checkpoint:
            warmup_epochs = 0

        self.scheduler = get_sqrt_schedule_with_warmup(
            self.opt,
            num_warmup_steps=len(self.data) * warmup_epochs,
            num_training_steps=len(self.data) * self.decay_epochs,
            num_cycles=1,
            min_decay=5e-2,
        )

        self.ema_params = copy.deepcopy(self.master_params)

        if torch.cuda.is_available():
            self.use_ddp = True
            print(get_device())
            self.ddp_model = DDP(
                self.model,
                device_ids=[get_device()],
                output_device=get_device(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
                # static_graph=True,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.model

        self.loaded_epoch = 0
        if load_checkpoint:
            files = glob.glob(os.path.join(self.checkpoint_path, "*.pt"))
            self.loaded_epoch = np.max([int(file[-12:].split(".")[0].split("_")[-1]) for file in files])

            file_name = f"model_ema_{self.loaded_epoch}.pt"
            logger.log(f"loading model from checkpoint: {file_name}...")
            # load best model for retraining
            checkpoint = load_state_dict(
                bf.join(self.checkpoint_path, file_name), map_location={"cuda:0": f"{get_device()}"}
            )
            self.ema_params = self._state_dict_to_master_params(checkpoint["ema"])
            self.model.load_state_dict(checkpoint["model"])
            if load_opt:
                self.opt.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["lr_schedule"])

    def run_loop(self):
        previous_loss = np.inf
        for epoch in range(self.loaded_epoch + 1, self.max_epochs + 1):
            self.data.sampler.set_epoch(epoch)
            self.eval_data.sampler.set_epoch(epoch)

            # train
            self.train_epoch(epoch)

            # evaluate
            current_loss = self.evaluate_epoch()

            # loss monitor
            if is_main_process():
                logger.log(f"Evaluation loss: {previous_loss:.5f} --> {current_loss:.5f}.")
            previous_loss = current_loss

            # save
            if epoch % self.save_epochs == 0:
                # early stop
                checkpoint = {
                    "model": self._master_params_to_state_dict(self.master_params),
                    "ema": self._master_params_to_state_dict(self.ema_params),
                    "optimizer": self.opt.state_dict(),
                    "lr_schedule": self.scheduler.state_dict(),
                }
                if is_main_process():
                    logger.log(f"Saving model after epoch {epoch}.")
                    torch.save(checkpoint, self.checkpoint_path + f"/model_ema_{epoch}.pt")
                dist.barrier()

        if is_main_process():
            logger.log("Training finished.")

    def train_epoch(self, epoch):
        self.ddp_model.train()
        # train
        n_batches = len(self.data)
        start_time = time.time()
        all_start_time = start_time
        self.opt.zero_grad()
        current_step = 0
        for i, inputs in enumerate(self.data):
            self.step += 1
            current_step += 1

            self.run_step(inputs)
            self.scheduler.step()

            # log
            if current_step % self.log_interval == 0:
                logger.dumpkvs()
                if is_main_process():
                    logger.log(
                        "Epoch {}, {:.1f}% took: {:.2f}s".format(epoch, 100 * i / n_batches, time.time() - start_time)
                    )
                start_time = time.time()

        logger.dumpkvs()
        if is_main_process():
            logger.log("Epoch {} took: {:.2f}s".format(epoch, time.time() - all_start_time))

    def evaluate_epoch(self):
        self.ddp_model.eval()

        return_loss = []
        for inputs in self.eval_data:
            return_loss.extend(self.forward_only(inputs))

        if is_main_process():
            logger.log("eval on validation set")
        logger.dumpkvs()

        gathered_loss = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_loss, return_loss)

        return torch.stack([torch.stack(loss) for loss in gathered_loss]).mean().numpy()

    def run_step(self, inputs):
        self.forward_backward(inputs)
        self.optimize_normal()
        self.log_step()

    def forward_only(self, inputs):
        return_loss = []
        src, tgt, src_ctx, tgt_cxt = inputs

        for i in range(0, src.shape[0], self.microbatch):
            src_micro = src[i : i + self.microbatch].to(get_device()).long()
            tgt_micro = tgt[i : i + self.microbatch].to(get_device()).long()
            src_ctx_micro = {k: v[i : i + self.microbatch].to(get_device()) for k, v in src_ctx.items()}
            tgt_ctx_micro = {k: v[i : i + self.microbatch].to(get_device()) for k, v in tgt_cxt.items()}

            last_batch = (i + self.microbatch) >= src.shape[0]
            t, weights = self.schedule_sampler.sample(src_micro.shape[0], get_device())

            with self.ddp_model.no_sync():
                grads = []
                # get the representation
                with torch.no_grad():
                    rep = self.ddp_model.module.decoder.forward_embedding(tgt_micro, tgt_ctx_micro).detach()
                rep.requires_grad_()

                # location
                self.opt.zero_grad()
                loss = self.ddp_model.module.get_loss_location(rep, tgt_micro)
                self.ddp_model.reducer.prepare_for_backward(loss)
                loss.mean().backward()
                grads.append(rep.grad.clone().detach())
                rep.grad.data.zero_()

                # mode
                self.opt.zero_grad()
                loss = self.ddp_model.module.get_loss_mode(rep, tgt_micro, tgt_ctx_micro)
                self.ddp_model.reducer.prepare_for_backward(loss)
                loss.mean().backward()
                grads.append(rep.grad.clone().detach())
                rep.grad.data.zero_()

                # duration
                self.opt.zero_grad()
                loss = self.ddp_model.module.get_loss_duration(rep, tgt_micro, tgt_ctx_micro)
                self.ddp_model.reducer.prepare_for_backward(loss)
                loss.mean().backward()
                grads.append(rep.grad.clone().detach())
                rep.grad.data.zero_()

                sol, _ = MinNormSolver.find_min_norm_element(grads)

                self.opt.zero_grad()

            with torch.no_grad():
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.ddp_model, src_micro, tgt_micro, src_ctx_micro, tgt_ctx_micro, t, sol
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                log_loss_dict(self.diff_steps, t, {f"eval_{k}": v * weights for k, v in losses.items()})
                return_loss.extend((losses["loss"] * weights).detach().cpu())

        return return_loss

    def forward_backward(self, inputs):
        current_device = get_device()
        src, tgt, src_ctx, tgt_cxt = inputs
        for i in range(0, src.shape[0], self.microbatch):
            src_micro = src[i : i + self.microbatch].to(get_device()).long()
            tgt_micro = tgt[i : i + self.microbatch].to(get_device()).long()
            src_ctx_micro = {k: v[i : i + self.microbatch].to(get_device()) for k, v in src_ctx.items()}
            tgt_ctx_micro = {k: v[i : i + self.microbatch].to(get_device()) for k, v in tgt_cxt.items()}

            last_batch = (i + self.microbatch) >= src.shape[0]
            t, weights = self.schedule_sampler.sample(src_micro.shape[0], current_device)
            # print(micro_cond.keys())

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_fp16):

                with self.ddp_model.no_sync():
                    grads = []
                    # get the representation
                    with torch.no_grad():
                        rep = self.ddp_model.module.decoder.forward_embedding(tgt_micro, tgt_ctx_micro).detach()
                    rep.requires_grad_()

                    # location
                    self.opt.zero_grad()
                    loss = self.ddp_model.module.get_loss_location(rep, tgt_micro)
                    self.ddp_model.reducer.prepare_for_backward(loss)
                    loss.mean().backward()
                    grads.append(rep.grad.clone().detach())
                    rep.grad.data.zero_()

                    # mode
                    self.opt.zero_grad()
                    loss = self.ddp_model.module.get_loss_mode(rep, tgt_micro, tgt_ctx_micro)
                    self.ddp_model.reducer.prepare_for_backward(loss)
                    loss.mean().backward()
                    grads.append(rep.grad.clone().detach())
                    rep.grad.data.zero_()

                    # duration
                    self.opt.zero_grad()
                    loss = self.ddp_model.module.get_loss_duration(rep, tgt_micro, tgt_ctx_micro)
                    self.ddp_model.reducer.prepare_for_backward(loss)
                    loss.mean().backward()
                    grads.append(rep.grad.clone().detach())
                    rep.grad.data.zero_()

                    sol, _ = MinNormSolver.find_min_norm_element(grads)

                    self.opt.zero_grad()

                compute_losses = functools.partial(
                    self.ddp_model, src_micro, tgt_micro, src_ctx_micro, tgt_ctx_micro, t, sol
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean() / self.grad_accum_steps
            log_loss_dict(self.diff_steps, t, {k: v * weights for k, v in losses.items()})

            loss.backward()

    def optimize_normal(self):
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

        self.log_grad_norm(total_norm)
        self.opt.step()
        self.opt.zero_grad()
        update_ema(self.ema_params, self.master_params, rate=self.ema_rate)

    def log_grad_norm(self, total_norm):
        logger.logkv_mean("grad_norm", total_norm)

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)
        logger.logkv("learning_rate", self.opt.param_groups[0]["lr"])

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if is_main_process():
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step):06d}.pt"
                print("writing to", bf.join(self.checkpoint_path, filename))

                with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f:  # DEBUG **
                    torch.save(state_dict, f)  # save locally

        save_checkpoint(0, self.master_params)
        save_checkpoint(self.ema_rate, self.ema_params)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.model.state_dict()
        for i, (name, _) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        return [state_dict[name] for name, _ in self.model.named_parameters()]


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def log_loss_dict(diff_steps, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diff_steps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def is_main_process():
    return dist.get_rank() == 0
