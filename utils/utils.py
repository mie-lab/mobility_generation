import os
import torch
import numpy as np
import pandas as pd

import json

import yaml
import random

from sklearn.preprocessing import OrdinalEncoder
from joblib import Parallel, delayed

from tqdm import tqdm

import pickle as pickle


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
    if config.split == "test":
        networkName += "_evaluate"

    log_dir = os.path.join(config.save_root, f"{networkName}_{str(time_now)}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
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
    sp.loc[sp["act_duration"] > 60 * 24 * 2 - 1, "act_duration"] = 60 * 24 * 2 - 1

    # split the datasets, user dependent 0.6, 0.2, 0.2
    train_data, vali_data, test_data = _split_dataset(sp)

    # encode unseen locations in validation and test into 0
    if all_locs is None:
        enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
            train_data["location_id"].values.reshape(-1, 1)
        )
        # add 2 to account for unseen locations (1) and to account for 0 padding
        train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
        vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
        test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

        return train_data, vali_data, test_data
    else:
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
        train_split = maxDay * 0.6
        vali_split = maxDay * 0.8

        df["Dataset"] = "test"
        df.loc[df["start_day"] < train_split, "Dataset"] = "train"
        df.loc[
            (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
            "Dataset",
        ] = "vali"

        return df

    totalData = totalData.groupby("user_id", group_keys=False).apply(getSplitDaysUser)

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
