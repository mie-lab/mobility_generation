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
from torch.optim import AdamW

torch.autograd.set_detect_anomaly(True)

import pickle as pickle

from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from utils import logger
from utils.dist_util import get_device, load_state_dict

from generative.diff.step_sample import LossAwareSampler


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
    ):
        self.model = model
        self.diff_steps = diff_steps
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
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
        param_groups = []
        name_group = ["lm_head", "token_embedding"]
        for name, parameter in self.model.named_parameters():
            if np.any([n in name for n in name_group]):
                param_groups.append({"params": [parameter], "lr": self.lr})
            else:
                param_groups.append({"params": [parameter], "lr": self.lr})

        self.opt = AdamW(param_groups, lr=self.lr, weight_decay=weight_decay, eps=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(growth_interval=3000, init_scale=256, enabled=self.use_fp16)
        # define learning rate schedule
        if load_checkpoint:
            warmup_epochs = 0
        if self.decay_epochs == 0:
            self.scheduler = get_constant_schedule_with_warmup(
                self.opt, num_warmup_steps=len(self.data) * warmup_epochs
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.opt,
                num_warmup_steps=len(self.data) * warmup_epochs,
                num_training_steps=len(self.data) * self.decay_epochs,
            )

        self.ema_params = copy.deepcopy(self.master_params)

        if torch.cuda.is_available():  # DEBUG **
            self.use_ddp = True
            print(get_device())
            self.ddp_model = DDP(
                self.model,
                device_ids=[get_device()],
                output_device=get_device(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. " "Gradients will not be synchronized properly!")
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
            self.scaler.load_state_dict(checkpoint["scaler"])
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
                    "scaler": self.scaler.state_dict(),
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
        with torch.no_grad():
            for i in range(0, src.shape[0], self.microbatch):
                src_micro = src[i : i + self.microbatch].to(get_device()).long()
                tgt_micro = tgt[i : i + self.microbatch].to(get_device()).long()
                src_ctx_micro = {k: v[i : i + self.microbatch].to(get_device()) for k, v in src_ctx.items()}
                tgt_ctx_micro = {k: v[i : i + self.microbatch].to(get_device()) for k, v in tgt_cxt.items()}

                last_batch = (i + self.microbatch) >= src.shape[0]
                t, weights = self.schedule_sampler.sample(src_micro.shape[0], get_device())
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.ddp_model, src_micro, tgt_micro, src_ctx_micro, tgt_ctx_micro, t
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

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                compute_losses = functools.partial(
                    self.ddp_model, src_micro, tgt_micro, src_ctx_micro, tgt_ctx_micro, t
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

                loss = (losses["loss"] * weights).mean()
                log_loss_dict(self.diff_steps, t, {k: v * weights for k, v in losses.items()})

            self.scaler.scale(loss).backward()

    def grad_clip(self):
        max_grad_norm = self.gradient_clipping  # 3.0
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.opt.clip_grad_norm(max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),  # amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )
            if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
                logger.log("nan encountered")

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            self.scaler.unscale_(self.opt)
            self.grad_clip()
        self._log_grad_norm()
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.use_fp16 and self.scaler._scale < 128:
            self.scaler._scale = torch.tensor(128).to(self.scaler._scale)

        self.opt.zero_grad()
        update_ema(self.ema_params, self.master_params, rate=self.ema_rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        # cnt = 0
        for p in self.master_params:
            if p.grad is not None:
                sqsum += (p.grad**2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))
        if self.use_fp16:
            logger.logkv("scaler scale", self.scaler._scale)

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
