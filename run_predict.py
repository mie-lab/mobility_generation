import argparse
import numpy as np
import random

import torch
import os

import pandas as pd
import datetime

import yaml

from easydict import EasyDict as edict

from loc_predict.processing import prepare_nn_dataset
from loc_predict.dataloader import get_dataloaders
from loc_predict.utils import get_models, get_trained_nets, get_test_result, init_save_path


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


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def single_run(train_loader, val_loader, test_loader, config, device, log_dir):
    result_ls = []

    # get models
    model = get_models(config, device)

    # train, returns validation performances
    model, perf = get_trained_nets(config, model, train_loader, val_loader, device, log_dir)
    result_ls.append(perf)

    # test, return test performances
    perf = get_test_result(config, model, test_loader, device)

    result_ls.append(perf)

    return result_ls


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
        df["duration"] = (df["duration"] * 60).round()
        return df

    sp = sp.groupby("user_id", group_keys=False).apply(_get_time_info)
    return sp


if __name__ == "__main__":
    setup_seed(0)

    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./loc_predict/config/mhsa.yml",
    )
    args = parser.parse_args()

    # initialization
    config = load_config(args.config)
    config = edict(config)

    # read and preprocess
    sp = pd.read_csv(os.path.join(config.temp_save_root, "sp.csv"), index_col="id")
    loc = pd.read_csv(os.path.join(config.temp_save_root, "locs_s2.csv"), index_col="id")
    sp = load_data(sp, loc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get data for nn, initialize the location and user number
    max_locations, max_users = prepare_nn_dataset(sp, config.temp_save_root)
    config["total_loc_num"] = int(max_locations + 1)
    config["total_user_num"] = int(max_users + 1)

    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # neural networks
    # possibility to enable multiple runs
    result_ls = []
    for i in range(1):
        # train, validate and test
        log_dir = init_save_path(config)
        # res_single contains the performance of validation and test of the current run
        res_single = single_run(train_loader, val_loader, test_loader, config, device, log_dir)
        result_ls.extend(res_single)

    # save results
    result_df = pd.DataFrame(result_ls)
    train_type = "default"
    filename = os.path.join(
        config.save_root,
        f"{config.dataset}_{config.networkName}_{train_type}_{str(int(datetime.datetime.now().timestamp()))}.csv",
    )
    result_df.to_csv(filename, index=False)
