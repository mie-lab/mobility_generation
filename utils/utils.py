import os
import torch
import numpy as np
import pandas as pd
import json
import yaml
import random

from joblib import Parallel, delayed
from easydict import EasyDict as edict
import pickle as pickle
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder

from diffusion.transformer_model import TransformerNetModel


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


def load_data(sp, loc):
    sp = sp.merge(loc.reset_index().drop(columns={"user_id"}), how="left", left_on="location_id", right_on="id")
    sp = sp.drop(columns={"location_id", "id", "center", "extent"})
    sp = sp.rename(columns={"s2_id": "location_id"})

    sp.index.name = "id"
    sp.reset_index(inplace=True)

    sp["started_at"] = pd.to_datetime(sp["started_at"], format="mixed", yearfirst=True, utc=True).dt.tz_localize(None)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"], format="mixed", yearfirst=True, utc=True).dt.tz_localize(None)

    def _get_time_info(df):
        min_day = pd.to_datetime(df["started_at"].min().date())

        # get the alighned time with act_duration
        df["temp_time"] = pd.NA
        df["temp_time"] = df["finished_at"].shift(1)
        df.loc[df.index[0], "temp_time"] = df["started_at"].iloc[0]

        df["start_day"] = (df["temp_time"] - min_day).dt.days
        df["start_min"] = df["temp_time"].dt.hour * 60 + df["temp_time"].dt.minute
        df["weekday"] = df["temp_time"].dt.weekday

        df = df.drop(columns="temp_time")
        return df

    sp = sp.groupby("user_id", group_keys=False).apply(_get_time_info)
    return sp


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(path):
    """
    Loads config file
    Args:
        path (str): path to the config file
    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for _, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def init_save_path(config, time_now):
    """define the path to save, and save the configuration file."""
    if config.networkName == "rnn" and config.attention:
        networkName = f"{config.dataset}_{config.networkName}_Attn"
    else:
        networkName = f"{config.dataset}_{config.networkName}"
    if config.split == "test":
        networkName += "_evaluate"

    log_dir = os.path.join(config.save_root, f"{networkName}_{str(time_now)}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "conf.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    return log_dir


def get_train_test(sp, all_locs=None):
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp.drop(columns={"started_at", "finished_at"}, inplace=True)

    # encoder user, 0 reserved for padding
    enc = OrdinalEncoder(dtype=np.int64)
    sp["user_id"] = enc.fit_transform(sp["user_id"].values.reshape(-1, 1)) + 1

    # truncate too long duration, >2 days to 2 days
    sp.loc[sp["act_duration"] > 60 * 24 * 2, "act_duration"] = 60 * 24 * 2

    # split the datasets, user dependent 0.7, 0.2, 0.1
    train_data, vali_data, test_data = _split_dataset(sp)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        all_locs["loc_id"].values.reshape(-1, 1)
    )
    # add 1 to account for 0 padding
    all_locs["loc_id"] = enc.transform(all_locs["loc_id"].values.reshape(-1, 1)) + 1

    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 1
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 1
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 1

    return train_data, vali_data, test_data, all_locs


def _split_dataset(totalData):
    """Split dataset into train, vali and test."""

    def getSplitDaysUser(df):
        """Split the dataset according to the tracked day of each user."""
        maxDay = df["start_day"].max()
        train_split = maxDay * 0.7
        vali_split = maxDay * 0.9

        df["Dataset"] = "test"
        df.loc[df["start_day"] < train_split, "Dataset"] = "train"
        df.loc[
            (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
            "Dataset",
        ] = "vali"

        return df

    totalData = totalData.groupby("user_id").apply(getSplitDaysUser, include_groups=False).reset_index()

    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # final cleaning
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data


def get_valid_start_end_idx(
    train_data, vali_data, test_data=None, print_progress=True, previous_day=7, return_test=True
):
    train_idx = _get_valid_sequence(train_data, print_progress=print_progress, previous_day=previous_day)
    vali_idx = _get_valid_sequence(vali_data, print_progress=print_progress, previous_day=previous_day)
    if return_test:
        test_idx = _get_valid_sequence(test_data, print_progress=print_progress, previous_day=previous_day)

        return train_idx, vali_idx, test_idx
    else:
        return train_idx, vali_idx


def _get_valid_sequence(input_df, print_progress=True, previous_day=7):
    def getValidSequenceUser(df, previous_day=7):
        id_ls = []
        df.reset_index(drop=True, inplace=True)

        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days

        for index, row in df.iterrows():
            # exclude the first records
            if row["diff_day"] < previous_day:
                continue

            curr_trace = df.iloc[: index + 1]
            curr_trace = curr_trace.loc[(curr_trace["start_day"] >= (row["start_day"] - previous_day))]

            # exclude series which contains too few records
            if len(curr_trace) > 2:
                id_ls.append([curr_trace["id"].values[0], curr_trace["id"].values[-1] + 1])

        return id_ls

    def applyParallel(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
        return Parallel(n_jobs=n_jobs)(
            delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
        )

    valid_user_ls = applyParallel(
        input_df.groupby("user_id"),
        getValidSequenceUser,
        n_jobs=-1,
        previous_day=previous_day,
        print_progress=print_progress,
    )
    return [item for sublist in valid_user_ls for item in sublist]
