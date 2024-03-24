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
        model_name=config.model_name,
        max_location=config.max_location,
    )

    betas = gaussian_diffusion.get_named_beta_schedule(config.noise_schedule, config.diffusion_steps)

    if not config.timestep_respacing:
        timestep_respacing = [config.diffusion_steps]

    diffusion = gaussian_diffusion.SpacedDiffusion(
        use_timesteps=gaussian_diffusion.space_timesteps(config.diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=config.rescale_timesteps,
        predict_xstart=config.predict_xstart,
    )

    return model, diffusion
