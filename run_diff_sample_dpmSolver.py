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

from functools import partial

from tqdm import tqdm
import numpy as np
import torch as th
from torch.cuda.amp import autocast
import torch.distributed as dist

from generative.dataloader import load_data_diffusion
from generative.diff.diff_utils import create_model_and_diffusion, denoised_fn_round
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

    print("### Sampling...on", config.split)

    ## load data
    data_valid = load_data_diffusion(batch_size=config.batch_size, shuffle=False, data_args=config, split=config.split)
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
            inputs = next(data_valid)
            all_test_data.append(inputs)

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

    if config.clamp:
        model_emb = (
            th.nn.Embedding(
                num_embeddings=config.max_location,
                embedding_dim=config.hidden_dim,
                _weight=model.token_embedding.weight.clone().cpu(),
            )
            .eval()
            .requires_grad_(False)
        )
        model_emb.to(dist_util.get_device())

        dpm_solver = DPM_Solver(
            model_fn,
            noise_schedule,
            algorithm_type="dpmsolver++",
            correcting_xt_fn=partial(denoised_fn_round, config, model_emb),
        )
    else:
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

    for inputs in tqdm(all_test_data):
        src, tgt, src_ctx, tgt_cxt = inputs
        src = src.to(dist_util.get_device()).long()
        tgt = tgt.to(dist_util.get_device()).long()

        encoder_out = model.encoder(src)

        # padding_mask B x T
        mask = tgt.ne(0)

        z_0 = model.decoder.forward_embedding(tgt)
        noise = th.randn_like(z_0) * config.rescaling_factor

        # adding or not does not influence the results
        # if model.mean_embed is not None:
        #     mean_embed = model.mean_embed.expand(x_noised.shape)
        #     mask = input_ids_mask.unsqueeze(dim=-1).expand(x_noised.shape).to(dist_util.get_device())
        #     x_noised += mean_embed * mask

        ## You can use steps = 10, 12, 15, 20, 25, 50, 100.
        ## Empirically, we find that steps in [10, 20] can generate quite good samples.
        ## And steps = 20 can almost converge.
        with autocast():
            x_sample = dpm_solver.sample(
                noise,
                encoder_out=encoder_out,
                steps=SOLVER_STEP,
                order=1,
                skip_type="time_uniform",
                method="multistep",
                padding_mask=mask,
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

        pred_ls = []
        true_ls = []
        input_ls = []

        for seq_pred, seq_src, seq_tgt in zip(cands, src, tgt):
            pred_ls.append(seq_pred.detach().cpu().numpy())

            true_ls.append(seq_src.detach().cpu().numpy())
            input_ls.append(seq_tgt.detach().cpu().numpy())

        for i in range(world_size):
            if i == rank:  # Write files sequentially
                fout = open(out_path, "a")
                for recov, src, tgt in zip(pred_ls, input_ls, true_ls):
                    print(
                        json.dumps({"recover": recov, "reference": tgt, "source": src}, cls=NumpyArrayEncoder),
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
