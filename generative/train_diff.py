import copy
import functools
import os

import blobfile as bf
import numpy as np
import time

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from utils import logger
from utils.dist_util import get_device, load_state_dict

import pickle as pickle

from utils.earlystopping import EarlyStopping

from generative.diff.step_sample import LossAwareSampler, UniformSampler
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        warmup_epochs=2,
        decay_epochs=100,
        early_stop_gamma,
        early_stop_patience,
        use_fp16=False,
        schedule_sampler=None,
        weight_decay=0.0,
        checkpoint_path="",
        gradient_clipping=-1.0,
        eval_data=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        self.log_interval = log_interval
        self.decay_epochs = decay_epochs
        self.use_fp16 = use_fp16
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.gradient_clipping = gradient_clipping

        self.step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

        self.checkpoint_path = checkpoint_path

        self.early_stop_patience = early_stop_patience

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)
        # define learning rate schedule
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
        # early stopping
        self.scheduler_ES = StepLR(self.opt, step_size=1, gamma=early_stop_gamma)
        self.ES = EarlyStopping(
            checkpoint_path,
            patience=early_stop_patience,
            main_process=is_main_process(),
            verbose=True,
            monitor="loss",
            delta=0.0001,
        )

        self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]

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

    def run_loop(self):
        early_stop_count = 0
        for epoch in range(1000):  # stop managed by ES
            self.data.sampler.set_epoch(epoch)
            self.eval_data.sampler.set_epoch(epoch)

            # train
            self.train_epoch(epoch, early_stop_count)

            # evaluate
            current_loss = self.evaluate_epoch()

            # early stop
            checkpoint = {
                "model": self._master_params_to_state_dict(self.ema_params[0]),
                "optimizer": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
            }

            self.ES(
                {"loss": current_loss},
                state_dict=checkpoint,
                save_name=f"ema_{self.ema_rate[0]}_{epoch}",
            )
            dist.barrier()
            if self.ES.early_stop:
                if is_main_process():
                    logger.log("=" * 50)
                    logger.log("Early stopping")
                    

                # only es for 2 times
                if early_stop_count == 2:
                    if is_main_process():
                        logger.log("Training finished.")
                    break

                if is_main_process():
                    logger.log(f"loading model from checkpoint: {self.ES.save_name}...")

                
                # load best model for retraining
                map_location = {"cuda:0": f"{get_device()}"}
                checkpoint = load_state_dict(
                    bf.join(self.checkpoint_path, self.ES.save_name + ".pt"), map_location=map_location
                )
                self.model.load_state_dict(checkpoint["model"])
                self.opt.load_state_dict(checkpoint["optimizer"])
                self.scaler.load_state_dict(checkpoint["scaler"])
                #
                early_stop_count += 1
                # reset
                self.ES.early_stop = False
                self.ES.counter = 0
                self.scheduler_ES.step()
                if is_main_process():
                    logger.log("Current lr: {:.6f}".format(self.opt.param_groups[0]["lr"]))

    def train_epoch(self, epoch, early_stop_count):
        self.ddp_model.train()
        # train
        n_batches = len(self.data)
        start_time = time.time()
        all_start_time = start_time
        self.opt.zero_grad()
        current_step = 0
        for i, (batch, cond) in enumerate(self.data):
            self.step += 1
            current_step += 1
            self.run_step(batch, cond)

            if not early_stop_count:
                self.scheduler.step()

            # log
            if current_step % self.log_interval == 0:
                logger.dumpkvs()

                if is_main_process():
                    logger.log(
                        "Epoch {}, {:.1f}% took: {:.2f}s".format(
                            epoch, 100 * i / n_batches, time.time() - start_time
                        )
                    )
                start_time = time.time()
        logger.dumpkvs()
        if is_main_process():
            logger.log("Epoch {} took: {:.2f}s".format(epoch + 1, time.time() - all_start_time))

    def evaluate_epoch(self):
        self.ddp_model.eval()

        return_loss = []
        for batch_eval, cond_eval in self.eval_data:
            return_loss.extend(self.forward_only(batch_eval, cond_eval))

        if is_main_process():
            logger.log("eval on validation set")
        logger.dumpkvs()

        gathered_loss = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_loss, return_loss)

        return torch.stack([torch.stack(loss) for loss in gathered_loss]).mean().numpy()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.optimize_normal()
        self.log_step()

    def forward_only(self, batch, cond):
        return_loss = []
        with torch.no_grad():
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i : i + self.microbatch].to(get_device())
                micro_cond = {k: v[i : i + self.microbatch].to(get_device()) for k, v in cond.items()}
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], get_device())
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                log_loss_dict(self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()})
                return_loss.extend((losses["loss"] * weights).detach().cpu())

        return return_loss

    def forward_backward(self, batch, cond):
        current_device = get_device()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(current_device)
            micro_cond = {k: v[i : i + self.microbatch].to(current_device) for k, v in cond.items()}
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], current_device)
            # print(micro_cond.keys())

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

            self.scaler.scale(loss).backward()

    def grad_clip(self):
        max_grad_norm = self.gradient_clipping  # 3.0
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.opt.clip_grad_norm(max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),  # amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            self.scaler.unscale_(self.opt)
            self.grad_clip()
        self._log_grad_norm()
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        # cnt = 0
        for p in self.master_params:
            # print(cnt, p) ## DEBUG
            # print(cnt, p.grad)
            # cnt += 1
            if p.grad is not None:
                sqsum += (p.grad**2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

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
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

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


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def is_main_process():
    return dist.get_rank() == 0
