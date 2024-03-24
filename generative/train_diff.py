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

from generative.diff.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)

import pickle as pickle


from metrics.evaluations import Metric
from utils.earlystopping import EarlyStopping

from generative.diff.step_sample import LossAwareSampler, UniformSampler
from transformers import get_linear_schedule_with_warmup

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


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
        max_epoch=100,
        warmup_epochs=2,
        early_stop_gamma,
        early_stop_patience,
        use_fp16=False,
        fp16_scale_growth=1e-3,
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
        self.max_epoch = max_epoch
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.gradient_clipping = gradient_clipping

        self.step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        self.checkpoint_path = checkpoint_path

        self.early_stop_patience = early_stop_patience

        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=weight_decay)
        # define learning rate schedule
        self.scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=len(self.data) * warmup_epochs,
            num_training_steps=len(self.data) * self.max_epoch,
        )
        # early stopping
        self.scheduler_ES = StepLR(self.opt, step_size=1, gamma=early_stop_gamma)
        self.ES = EarlyStopping(
            checkpoint_path, patience=early_stop_patience, verbose=True, monitor="loss", delta=0.0001
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

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        early_stop_count = 0
        for epoch in range(self.max_epoch):
            self.data.sampler.set_epoch(epoch)
            self.eval_data.sampler.set_epoch(epoch)

            # train
            self.train_epoch(epoch, early_stop_count)

            # evaluate
            current_loss = self.evaluate_epoch()

            # early stop
            self.ES(
                {"loss": current_loss},
                self._master_params_to_state_dict(self.master_params),
                save_name=f"model_{epoch}",
            )
            if self.ES.early_stop:
                if dist.get_rank() == 0:
                    print("=" * 50)
                    print("Early stopping")
                    print("Current lr: {:.6f}".format(self.opt.param_groups[0]["lr"]))
                    if early_stop_count == 2:
                        if dist.get_rank() == 0:
                            print("Training finished.")
                        break

                    logger.log(f"loading model from checkpoint: {self.ES.save_name}...")

                dist.barrier()
                map_location = {"cuda:0": f"{get_device()}"}
                self.model.load_state_dict(
                    load_state_dict(bf.join(self.checkpoint_path, self.ES.save_name + ".pt"), map_location=map_location)
                )
                #
                early_stop_count += 1
                self.ES.early_stop = False
                self.ES.counter = 0
                self.scheduler_ES.step()

    def train_epoch(self, epoch, early_stop_count):
        # train
        n_batches = len(self.data)
        start_time = time.time()
        self.opt.zero_grad()
        for i, (batch, cond) in enumerate(self.data):
            self.run_step(batch, cond)

            if not early_stop_count:
                self.scheduler.step()

            # log
            if (self.step) % self.log_interval == 0:
                logger.dumpkvs()
                if dist.get_rank() == 0:
                    print(
                        "Epoch {}, {:.1f}% took: {:.2f}s".format(
                            epoch + 1, 100 * i / n_batches, time.time() - start_time
                        )
                    )
                start_time = time.time()

            self.step += 1

    def evaluate_epoch(self):
        self.ddp_model.eval()

        return_loss = 0
        for batch_eval, cond_eval in self.eval_data:
            return_loss += self.forward_only(batch_eval, cond_eval)
        print("eval on validation set")
        logger.dumpkvs()
        return return_loss

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_only(self, batch, cond):
        return_loss = []
        with torch.no_grad():
            zero_grad(self.model_params)
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
                return_loss.extend(losses["loss"].detach().cpu().numpy().tolist())

        return np.mean(return_loss)

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
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

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            self.opt.zero_grad()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            if self.use_fp16:
                loss_scale = 2**self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not torch.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2**self.lg_loss_scale))
        self._log_grad_norm()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def grad_clip(self):
        # print('doing gradient clipping')
        max_grad_norm = self.gradient_clipping  # 3.0
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.opt.clip_grad_norm(max_grad_norm)
        # else:
        #     assert False
        # elif hasattr(self.model, "clip_grad_norm_"):
        #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #     self.model.clip_grad_norm_(args.max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),  # amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            self.grad_clip()
        self._log_grad_norm()
        self.opt.step()
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
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
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
        if self.use_fp16:
            master_params = unflatten_master_params(list(self.model.parameters()), master_params)  # DEBUG **
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


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
