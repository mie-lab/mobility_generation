import argparse
import os
import json
from json import JSONEncoder
from easydict import EasyDict as edict

import torch

import torch.distributed as dist

import numpy as np
import datetime
import time
from tqdm import tqdm

from diffusion.dataloader import load_data
from utils import dist_util, logger
from utils.utils import setup_seed, load_config, init_save_path, create_model

from time import perf_counter as timer


@torch.no_grad()
def main():
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./config/diff_sample_geolife.yml",
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
    data_valid = load_data(
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

    time_ls = []
    device = dist_util.get_device()
    for inputs in tqdm(all_test_data):
        src, tgt, src_ctx, tgt_cxt = inputs
        src = src.to(device).long()
        tgt = tgt.to(device).long()
        src_ctx = {k: v.to(device) for k, v in src_ctx.items()}
        tgt_cxt = {k: v.to(device) for k, v in tgt_cxt.items()}

        start = timer()
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

        return_dict = model.forward_output_layer(prev_z_0_hat)

        time_ls.append((timer() - start) / len(return_dict["tokens"]))

        res_dict_ls = []
        for i, (seq_pred, seq_src, seq_tgt) in enumerate(zip(return_dict["tokens"], src, tgt)):

            res_dict = {
                "recover": seq_pred.detach().cpu().numpy(),
                "target": seq_tgt.detach().cpu().numpy(),
                "source": seq_src.detach().cpu().numpy(),
            }

            if config.if_include_duration:
                res_dict["duration"] = np.round(return_dict["durations"][i].detach().cpu().numpy(), 3)
                res_dict["time"] = np.round(return_dict["time"][i].detach().cpu().numpy(), 3)

                tgt_dur = tgt_cxt["duration"][i].detach().cpu().numpy()
                tgt_dur[tgt_dur != 0] = np.round(((tgt_dur[tgt_dur != 0] + 1) / 2 * 2880), 0)
                res_dict["tgt_dur"] = tgt_dur

                src_dur = src_ctx["duration"][i].detach().cpu().numpy()
                src_dur[src_dur != 0] = np.round(((src_dur[src_dur != 0] + 1) / 2 * 2880), 0)
                res_dict["src_dur"] = src_dur

                tgt_time = tgt_cxt["time"][i].detach().cpu().numpy()
                tgt_time[tgt_time != 0] = (tgt_time[tgt_time != 0] + 1) / 2 * 1440
                res_dict["tgt_time"] = np.round(tgt_time, 0)

                seq_time = src_ctx["time"][i].detach().cpu().numpy()
                seq_time[seq_time != 0] = (seq_time[seq_time != 0] + 1) / 2 * 1440
                res_dict["seq_time"] = np.round(seq_time, 0)

            if config.if_include_mode:
                res_dict["mode"] = return_dict["mode"][i].detach().cpu().numpy()
                res_dict["tgt_mode"] = tgt_cxt["mode"][i].detach().cpu().numpy()
                res_dict["src_mode"] = src_ctx["mode"][i].detach().cpu().numpy()

            res_dict_ls.append(res_dict)

        for i in range(world_size):
            if i == rank:  # Write files sequentially
                fout = open(out_path, "a")
                for res_dict in res_dict_ls:
                    print(json.dumps(res_dict, cls=NumpyArrayEncoder), file=fout)
                fout.close()
            dist.barrier()

    print(np.mean(np.array(time_ls)), np.std(np.array(time_ls)))
    print("### Total takes {:.2f}s .....".format(time.time() - start_t))
    print(f"### Written the decoded output to {out_path}")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


if __name__ == "__main__":
    main()
