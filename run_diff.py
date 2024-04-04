"""
Train a diffusion model on images.
"""

import argparse
import datetime
import json
import os

import numpy as np
import wandb
from easydict import EasyDict as edict

from generative.dataloader import load_data_diffusion
from generative.diff.diff_utils import create_model_and_diffusion
from generative.diff.step_sample import create_named_schedule_sampler
from generative.train_diff import TrainLoop
from utils import logger
from utils.dist_util import get_device, setup_dist
from utils.utils import init_save_path, load_config, setup_seed

### custom your wandb setting here ###
os.environ["WANDB_API_KEY"] = "7c18d47ede09b76c8d8e7d861b930edc81c2b0e8"
os.environ["WANDB_MODE"] = "online"


def main():
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./config/diff.yml",
    )
    parser.add_argument("--local-rank", type=int)
    args = parser.parse_args()

    # initialization
    config = load_config(args.config)
    config = edict(config)

    setup_seed(config.seed)
    setup_dist()

    time_now = int(datetime.datetime.now().timestamp())
    log_dir = init_save_path(config, time_now=time_now)

    logger.configure(dir=log_dir)
    logger.log("### Creating data loader...")

    data_train = load_data_diffusion(batch_size=config.batch_size, data_args=config, shuffle=True)
    data_valid = load_data_diffusion(batch_size=config.batch_size, data_args=config, split="valid", shuffle=False)

    logger.log("### Creating model and diffusion...")
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    config.device = get_device()
    model, diffusion = create_model_and_diffusion(config)
    model.to(get_device())  # DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    schedule_sampler = create_named_schedule_sampler(config.schedule_sampler, diffusion)

    config.device = ""
    if ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0):
        logger.log("#" * 30, "size of location", config.max_location)
        logger.log(f"### The parameter count is {pytorch_total_params}")
        logger.log("### Saving the hyperparameters")

        with open(f"{log_dir}/training_args.json", "w") as f:
            json.dump(config.__dict__, f, indent=2)

        wandb.init(
            project="DiffuSeq",
            dir=log_dir,
            name=f"{config.wandb_name}_{str(time_now)}",
        )
        wandb.config.update(config.__dict__, allow_val_change=True)

        logger.log("### Training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_train,
        batch_size=config.batch_size,
        microbatch=config.microbatch,
        lr=config.lr,
        ema_rate=config.ema_rate,
        log_interval=config.log_interval,
        use_fp16=config.use_fp16,
        schedule_sampler=schedule_sampler,
        weight_decay=config.weight_decay,
        early_stop_gamma=config.early_stop_gamma,
        early_stop_patience=config.early_stop_patience,
        decay_epochs=config.decay_epochs,
        warmup_epochs=config.warmup_epochs,
        checkpoint_path=log_dir,
        gradient_clipping=config.gradient_clipping,
        eval_data=data_valid,
    ).run_loop()


if __name__ == "__main__":
    main()
