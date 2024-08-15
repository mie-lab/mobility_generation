import argparse
import os
import json
from json import JSONEncoder
from functools import partial
from easydict import EasyDict as edict

import torch

import torch.distributed as dist
from torch.cuda.amp import autocast

import numpy as np
import datetime
import time
from tqdm import tqdm

from generative.diff.diff_utils import create_model
from generative.dataloader import load_data_diffusion
from utils import dist_util, logger
from utils.utils import setup_seed, load_config, init_save_path


@torch.no_grad()
def main():
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./config/diff_sample.yml",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)
    # load configurations.
    config_path = os.path.join(os.path.join(config.model_path, "training_args.json"))
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, "rb") as f:
        training_args = json.load(f)
    config = {**training_args, **config}
    config = edict(config)

    setup_seed(config.seed)
    dist_util.setup_dist()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    time_now = int(datetime.datetime.now().timestamp())
    log_dir = init_save_path(config, time_now=time_now)

    logger.configure(dir=log_dir)

    logger.log("### Creating model and diffusion...")
    config.device = dist_util.get_device()
    model = create_model(config)

    checkpoint = dist_util.load_state_dict(
        os.path.join(config.model_path, config.trained_model_name), map_location="cpu"
    )
    model.load_state_dict(checkpoint["ema"])

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"### The parameter count is {pytorch_total_params}")

    model.eval().requires_grad_(False).to(dist_util.get_device())

    print("### Sampling...on", config.split)

    ## load data
    data_valid = load_data_diffusion(
        batch_size=config.batch_size,
        shuffle=False,
        data_args=config,
        split=config.split,
    )
    data_valid = iter(data_valid)

    start_t = time.time()

    out_path = os.path.join(log_dir, f"seed{config.seed}_step{config.decoding_steps}.json")

    all_test_data = []

    try:
        while True:
            inputs = next(data_valid)
            all_test_data.append(inputs)

    except StopIteration:
        print("### End of reading iteration...")

    device = dist_util.get_device()
    for inputs in tqdm(all_test_data):
        src, tgt, src_ctx, tgt_cxt = inputs
        src = src.to(device).long()
        tgt = tgt.to(device).long()
        src_ctx = {k: v.to(device) for k, v in src_ctx.items()}
        tgt_cxt = {k: v.to(device) for k, v in tgt_cxt.items()}

        encoder_out = model.encoder(src, context=src_ctx)

        # padding_mask B x T
        mask = torch.ones_like(tgt) == 1

        # initialize
        z_0 = model.decoder.forward_embedding(tgt, tgt_cxt=tgt_cxt)
        z_t = torch.randn_like(z_0) * config.decoding_rescaling_factor
        z_t = z_t.to(encoder_out["encoder_out"])

        # self-conditioning
        prev_z_0_hat = torch.zeros_like(z_t)
        for step in list(range(config.decoding_steps))[::-1]:
            z_t, prev_z_0_hat = model.forward_decoder(z_t, step, mask, encoder_out, prev_z_0_hat)

        tokens, durations, modes, scores = model.forward_output_layer(prev_z_0_hat)

        res_dict_ls = []

        for seq_pred, pred_duration, pred_mode, seq_src, seq_tgt, seq_time, src_dur, src_mode, tgt_dur, tgt_mode in zip(
            tokens,
            durations,
            modes,
            src,
            tgt,
            src_ctx["time"],
            src_ctx["duration"],
            src_ctx["mode"],
            tgt_cxt["duration"],
            tgt_cxt["mode"],
        ):
            res_dict = {
                "recover": seq_pred.detach().cpu().numpy(),
                "target": seq_tgt.detach().cpu().numpy(),
                "source": seq_src.detach().cpu().numpy(),
                "seq_time": seq_time.detach().cpu().numpy(),
                "duration": pred_duration.detach().cpu().numpy(),
                "mode": pred_mode.detach().cpu().numpy(),
                "tgt_dur": ((tgt_dur + 1) / 2 * 2880).detach().cpu().numpy(),
                "tgt_mode": tgt_mode.detach().cpu().numpy(),
                "src_dur": src_dur.detach().cpu().numpy(),
                "src_mode": src_mode.detach().cpu().numpy(),
            }

            res_dict_ls.append(res_dict)

        for i in range(world_size):
            if i == rank:  # Write files sequentially
                fout = open(out_path, "a")
                for res_dict in res_dict_ls:
                    print(json.dumps(res_dict, cls=NumpyArrayEncoder), file=fout)
                fout.close()
            dist.barrier()

    print("### Total takes {:.2f}s .....".format(time.time() - start_t))
    print(f"### Written the decoded output to {out_path}")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


if __name__ == "__main__":
    main()
