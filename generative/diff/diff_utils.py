import os
import torch
import numpy as np
import pandas as pd

import json

import yaml
import random


from generative.diff import transformer_model, gaussian_diffusion

import pickle as pickle


def create_model_and_diffusion(config):
    model = transformer_model.TransformerNetModel(
        input_dims=config.hidden_dim,
        hidden_t_dim=config.hidden_t_dim,
        dropout=config.dropout,
        num_encoder_layers=config.num_encoder_layers,
        max_location=config.max_location,
    )

    betas = gaussian_diffusion.get_named_beta_schedule(config.noise_schedule, config.diffusion_steps)

    if not config.timestep_respacing:
        timestep_respacing = [config.diffusion_steps]

    diffusion = gaussian_diffusion.SpacedDiffusion(
        use_timesteps=space_timesteps(config.diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=config.rescale_timesteps,
        predict_xstart=config.predict_xstart,
    )

    return model, diffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def get_efficient_knn(model_emb, text_emb):
    emb_norm = (model_emb**2).sum(-1).view(-1, 1)  # vocab
    text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
    arr_norm = (text_emb**2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
    # print(emb_norm.shape, arr_norm.shape)
    dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(model_emb, text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
    dist = torch.clamp(dist, 0.0, np.inf)
    # print(dist.shape)
    topk_out = torch.topk(-dist, k=1, dim=0)
    return topk_out.values, topk_out.indices


def denoised_fn_round(args, model, old_embed, t):
    # print(text_emb.shape) # bsz, seqlen, dim
    model_emb = model.weight  # input_embs
    # print(t)
    old_shape = old_embed.shape
    old_device = old_embed.device

    if len(old_embed.shape) > 2:
        old_embed = old_embed.reshape(-1, old_embed.size(-1))
    else:
        old_embed = old_embed
    # clamp to the nearest word embedding
    val, indices = get_efficient_knn(model_emb, old_embed.to(model_emb.device))
    rounded_tokens = indices[0]
    # get the new (mapped) word embedding
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)

    return new_embeds
