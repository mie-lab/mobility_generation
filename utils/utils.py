import os
import torch
import numpy as np
import pandas as pd

import json

import yaml
import random


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

        df["start_day"] = (df["started_at"] - min_day).dt.days
        df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
        df["weekday"] = df["started_at"].dt.weekday
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

    log_dir = os.path.join(config.save_root, f"{networkName}_{str(time_now)}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "conf.json"), "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    return log_dir
