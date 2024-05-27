"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

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


from generative.diff.diff_utils import create_model_and_diffusion, denoised_fn_round
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

    setup_seed(config.seed)
    dist_util.setup_dist()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    time_now = int(datetime.datetime.now().timestamp())
    log_dir = init_save_path(config, time_now=time_now)

    logger.configure(dir=log_dir)

    # load configurations.
    config_path = os.path.join(os.path.join(config.model_path, "training_args.json"))
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, "rb") as f:
        training_args = json.load(f)
    training_args["batch_size"] = config.batch_size
    training_args["dataset_variation"] = config.dataset_variation
    training_args["data_dir"] = config.data_dir
    training_args["pre_train_embed"] = config.pre_train_embed
    training_args["save_root"] = config.save_root
    training_args["split"] = config.split
    CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])
    config.__dict__.update(training_args)
    config.device = f"cuda:{CUDA_VISIBLE_DEVICES}"

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(config)

    checkpoint = dist_util.load_state_dict(
        os.path.join(config.model_path, config.trained_model_name), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model"])

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"### The parameter count is {pytorch_total_params}")

    model.eval().requires_grad_(False).to(dist_util.get_device())

    model_emb = (
        torch.nn.Embedding(
            num_embeddings=config.max_location,
            embedding_dim=config.hidden_dim,
            _weight=model.token_embedding.weight.clone().cpu(),
        )
        .eval()
        .requires_grad_(False)
    )

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

    out_path = os.path.join(log_dir, f"seed{config.seed}_step{config.clamp_step}.json")

    all_test_data = []

    idx = 0
    try:
        while True:
            _, cond = next(data_valid)
            # print(batch.shape)
            if idx % world_size == rank:  # Split data per nodes
                all_test_data.append(cond)
            idx += 1

    except StopIteration:
        print("### End of reading iteration...")

    model_emb.to(dist_util.get_device())

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # Dummy data for Remainder : for dist.barrier()

    if rank == 0:
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    device = dist_util.get_device()
    for cond in iterator:
        if not cond:  # Barrier for Remainder
            for i in range(world_size):
                dist.barrier()
            continue

        input_ids = cond.pop("input_ids").to(device)
        x_start = model.get_embeds(input_ids)

        # padding_mask
        max_len = input_ids.shape[1]
        lens = cond.pop("len").to(device).int()
        padding_mask = (torch.arange(max_len).expand(len(lens), max_len).to(device) < lens.unsqueeze(1)) * 1

        input_ids_mask = cond.pop("input_mask").to(device)
        input_ids_mask_ori = input_ids_mask

        context = {}
        # noise input_xys
        if "input_xys" in cond:
            input_xys = cond.pop("input_xys").to(device).float()
            noise = torch.randn_like(input_xys)
            xy_mask = torch.broadcast_to(input_ids_mask.unsqueeze(dim=-1), input_xys.shape).to(device)
            context["xy"] = torch.where(xy_mask == 0, input_xys, noise)

        # noise x_start
        noise = torch.randn_like(x_start)
        x_mask = torch.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(device)
        x_noised = torch.where(x_mask == 0, x_start, noise)

        model_kwargs = {}

        if config.step == config.diffusion_steps:
            config.use_ddim = False
            step_gap = 1
        else:
            config.use_ddim = True
            step_gap = config.diffusion_steps // config.step

        sample_fn = diffusion.p_sample_loop

        # [batch, seq_len, hidden_dim]
        curr_seq_len = x_start.shape[1]
        sample_shape = (x_start.shape[0], curr_seq_len, config.hidden_dim)

        with autocast():
            samples = sample_fn(
                model,
                context,
                sample_shape,
                noise=x_noised,
                clip_denoised=config.clip_denoised,
                denoised_fn=partial(denoised_fn_round, config, model_emb),
                model_kwargs=model_kwargs,
                top_p=config.top_p,
                clamp_step=config.clamp_step,
                clamp_first=False,
                mask=input_ids_mask,
                padding_mask=padding_mask,
                x_start=x_start,
                gap=step_gap,
            )
        # only get the latest timestep ()
        sample = samples[-1]

        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        cands = torch.topk(logits, k=1, dim=-1).indices

        loc_ls_pred = []
        loc_ls_true = []
        loc_ls_input = []

        for seq_pred, seq_ref, input_mask in zip(cands, input_ids, input_ids_mask_ori):
            len_x = int(curr_seq_len - sum(input_mask).tolist())
            loc_ls_pred.append(seq_pred[len_x:].detach().cpu().numpy())

            loc_ls_true.append(seq_ref[len_x:].detach().cpu().numpy())
            loc_ls_input.append(seq_ref[:len_x].detach().cpu().numpy())

        for i in range(world_size):
            if i == rank:  # Write files sequentially
                fout = open(out_path, "a")
                for recov, ref, src in zip(loc_ls_pred, loc_ls_true, loc_ls_input):
                    print(
                        json.dumps({"recover": recov, "reference": ref, "source": src}, cls=NumpyArrayEncoder),
                        file=fout,
                    )
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
