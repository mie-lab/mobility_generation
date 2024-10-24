import json
import os
import pickle as pickle
import random

import numpy as np
import pandas as pd
import torch
import yaml

from easydict import EasyDict as edict


from .transformer_model import TransformerNetModel


def create_model(config):
    model_args = {
        "input_dims": config.input_dims,
        "num_layers": config.num_layers,
        "max_location": config.max_location,
        "max_mode": config.max_mode,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "dropout": config.dropout,
        "if_embed_context": config.if_embed_context,
        "poi_dim": config.poi_dim,
        "if_include_duration": config.if_include_duration,
        "if_include_mode": config.if_include_mode,
        "device": config.device,
    }
    model_args = edict(model_args)
    diff_args = {
        "noise_schedule": config.noise_schedule,
        "diffusion_steps": config.diffusion_steps,
        "rescaling_factor": config.rescaling_factor,
        "rescale_timesteps": config.rescale_timesteps,
        "rounding_loss": config.rounding_loss,
        "self_cond": config.self_cond,
        "decoding_steps": config.decoding_steps,
        "decoding_noise_schedule": config.decoding_noise_schedule,
        "decoding_rescaling_factor": config.decoding_rescaling_factor,
        "clamping": config.clamping,
    }
    diff_args = edict(diff_args)

    model = TransformerNetModel(model_args=model_args, diff_args=diff_args)

    return model


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


def denoised_fn_round(args, model, old_embed, t, step=None):
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
    _, indices = get_efficient_knn(model_emb, old_embed.to(model_emb.device))
    rounded_tokens = indices[0].view(old_shape[:-1]).to(old_device)
    new_embeds = model(rounded_tokens)

    return new_embeds, rounded_tokens


def get_weights(model, args):
    if hasattr(model, "transformer"):
        input_embs = model.transformer.wte  # input_embs
        down_proj = model.down_proj
        model_emb = down_proj(input_embs.weight)
        print(model_emb.shape)
        model = torch.nn.Embedding(model_emb.size(0), model_emb.size(1))
        print(args.emb_scale_factor)
        model.weight.data = model_emb * args.emb_scale_factor

    elif hasattr(model, "weight"):
        pass
    else:
        assert NotImplementedError

    model.weight.requires_grad = False
    return model
