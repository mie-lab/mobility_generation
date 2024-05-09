"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os


import time
import datetime

import json
from json import JSONEncoder
from easydict import EasyDict as edict

from tqdm import tqdm
import numpy as np
import torch as th
from torch.cuda.amp import autocast
import torch.distributed as dist

from generative.dataloader import load_data_diffusion
from generative.diff.diff_utils import create_model_and_diffusion
from generative.diff.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

from utils import dist_util, logger
from utils.utils import setup_seed, load_config, init_save_path


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
    training_args["pre_train_embed"] = config.pre_train_embed
    training_args["dataset_variation"] = config.dataset_variation
    training_args["data_dir"] = config.data_dir
    training_args["save_root"] = config.save_root
    training_args["split"] = config.split
    CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])
    config.__dict__.update(training_args)
    config.device = f"cuda:{CUDA_VISIBLE_DEVICES}"

    logger.log("### Creating model and diffusion...")
    # args.denoise_rate = 0.0
    print("#" * 10)
    model, diffusion = create_model_and_diffusion(config)

    checkpoint = dist_util.load_state_dict(
        os.path.join(config.model_path, config.trained_model_name), map_location="cpu"
    )
    model.load_state_dict(checkpoint["ema"])

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"### The parameter count is {pytorch_total_params}")

    model.to(dist_util.get_device())
    model.eval()

    # model_emb = torch.nn.Embedding(num_embeddings=config.max_location, embedding_dim=config.hidden_dim)
    # model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    print("### Sampling...on", config.split)

    ## load data
    data_valid = load_data_diffusion(
        batch_size=config.batch_size,
        shuffle=False,
        data_args=config,
        split=config.split,
        # model_emb=model_emb.cpu(),  # using the same embedding wight with training data
    )
    data_valid = iter(data_valid)

    start_t = time.time()

    # batch, cond = next(data_valid)
    # print(batch.shape)

    SOLVER_STEP = config.step

    out_path = os.path.join(log_dir, f"seed{config.seed}_solverstep{SOLVER_STEP}.json")
    # fout = open(out_path, 'a')

    all_test_data = []

    try:
        while True:
            _, cond = next(data_valid)
            all_test_data.append(cond)

    except StopIteration:
        print("### End of reading iteration...")

    print("Start from ...", config.start_n)
    all_test_data = all_test_data[config.start_n :]

    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=th.from_numpy(diffusion.betas))

    ## 2. Convert your discrete-time `model` to the continuous-time
    ## noise prediction model. Here is an example for a diffusion model
    ## `model` with the noise prediction type ("noise")
    model_kwargs = {}
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="x_start",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
        guidance_type="uncond",
    )

    ## 3. Define dpm-solver and sample by multistep DPM-Solver.
    ## (We recommend multistep DPM-Solver for conditional sampling)
    ## You can adjust the `steps` to balance the computation
    ## costs and the sample quality.

    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

    for cond in tqdm(all_test_data):
        input_ids_x = cond.pop("input_ids").to(dist_util.get_device())

        x_start = model.get_embeds(input_ids_x)

        # padding_mask
        max_len = input_ids_x.shape[1]
        lens = cond.pop("len").to(dist_util.get_device()).int()
        padding_mask = (
            th.arange(max_len).expand(len(lens), max_len).to(dist_util.get_device()) < lens.unsqueeze(1)
        ) * 1
        # padding_mask = (input_ids_x != 0) * 1

        input_ids_mask = cond.pop("input_mask")
        input_ids_mask_ori = input_ids_mask

        # noise input_xys
        context = {}
        if "input_xys" in cond:
            input_xys = cond.pop("input_xys").to(dist_util.get_device()).float()
            zeros = th.zeros_like(input_xys)
            xy_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), input_xys.shape).to(dist_util.get_device())
            context["xy"] = th.where(xy_mask == 0, input_xys, zeros)

        if "input_poi" in cond:
            input_poi = cond.pop("input_poi").to(dist_util.get_device()).float()

            noise = th.randn_like(input_poi)
            poi_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), input_poi.shape).to(dist_util.get_device())
            context["poi"] = th.where(poi_mask == 0, input_poi, noise)

        # noise x_start
        noise = th.randn_like(x_start)
        x_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.get_device())
        x_noised = th.where(x_mask == 0, x_start, noise)

        # if model.mean_embed is not None:
        #     mean_embed = model.mean_embed.expand(x_noised.shape)
        #     mask = input_ids_mask.unsqueeze(dim=-1).expand(x_noised.shape).to(dist_util.get_device())
        #     x_noised += mean_embed * mask

        ## You can use steps = 10, 12, 15, 20, 25, 50, 100.
        ## Empirically, we find that steps in [10, 20] can generate quite good samples.
        ## And steps = 20 can almost converge.
        with autocast():
            x_sample = dpm_solver.sample(
                x_noised,
                x_context=context,
                steps=SOLVER_STEP,
                order=2,
                skip_type="time_uniform",
                method="multistep",
                input_mask=input_ids_mask,
                x_start=x_start,
                padding_mask=padding_mask,
            )
        # print(x_sample[0].shape) # samples for each step [128, 128]

        sample = x_sample
        gathered_samples = [th.zeros_like(sample) for _ in range(world_size)]
        dist.all_gather(gathered_samples, sample)
        all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

        # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

        arr = np.concatenate(all_sentence, axis=0)
        x_t = th.tensor(arr).cuda()
        # print('decoding for seq2seq', )
        # print(arr.shape)

        reshaped_x_t = x_t
        logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1).indices

        loc_ls_pred = []
        loc_ls_true = []
        loc_ls_input = []

        curr_seq_len = x_start.shape[1]
        for seq_pred, seq_ref, input_mask in zip(cands, input_ids_x, input_ids_mask_ori):
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
