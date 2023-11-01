import argparse
import numpy as np
import random

import torch
import os

import pandas as pd
import datetime

import yaml

from easydict import EasyDict as edict

from loc_predict.dataloader import get_dataloaders, _get_train_test
from loc_predict.utils import get_models, get_trained_nets, get_test_result, init_save_path, get_generated_sequences
from loc_predict.models.markov import markov_transition_prob
from utils.utils import load_data


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


if __name__ == "__main__":
    setup_seed(0)

    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./config/mhsa.yml",
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

    # get dataloaders
    train_loader, val_loader, test_loader, max_locations, max_users = get_dataloaders(sp, config)
    config["total_loc_num"] = int(max_locations + 1)
    config["total_user_num"] = int(max_users + 1)

    if "mhsa" in args.config:  # neural networks
        if not config.use_pretrain:  # for training
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
        else:  # for generation
            model = get_models(config, device)
            model.load_state_dict(torch.load(os.path.join(config.save_root, config.pretrain_filepath, "checkpoint.pt")))

            generated_df = get_generated_sequences(config, model, test_loader, device)

            filename = os.path.join(
                config.save_root,
                f"{config.dataset}_{config.networkName}_generation_{str(int(datetime.datetime.now().timestamp()))}.csv",
            )
            generated_df.to_csv(filename, index=True)

    elif "markov" in args.config:  # markov model
        train_data, vali_data, test_data = _get_train_test(sp)

        # construct markov matrix based on train and validation dataset
        train_vali_data = pd.concat([train_data, vali_data])

        transition_df = (
            train_vali_data.groupby(["user_id"]).apply(markov_transition_prob, n=config.n_dependence).reset_index()
        )

        groupby_transition = transition_df.groupby("user_id")

        generated_df = get_generated_sequences(config, groupby_transition, test_loader)
        filename = os.path.join(
            config.save_root,
            f"{config.dataset}_{config.networkName}_generation_{str(int(datetime.datetime.now().timestamp()))}.csv",
        )
        generated_df.to_csv(filename, index=True)

    else:
        raise AttributeError("Prediction method not implemented.")
