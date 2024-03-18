import os
import numpy as np
import pickle as pickle
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
import torch

from joblib import Parallel, delayed

from sklearn.preprocessing import OrdinalEncoder

import trackintel as ti


class traj_dataset(torch.utils.data.Dataset):
    def __init__(self, input_data, config, valid_start_end_idx):
        self.data = input_data

        self.if_embed_poi = config.if_embed_poi
        self.valid_start_end_idx = valid_start_end_idx
        self.len = len(valid_start_end_idx)

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.len

    def __getitem__(self, idx):
        start_end_idx = self.valid_start_end_idx[idx]
        selected = self.data.iloc[start_end_idx[0] : start_end_idx[1]]

        return_dict = {}
        y_dict = {}

        loc_seq = selected["location_id"].values
        # [sequence_len]
        x = torch.tensor(loc_seq[:-1])
        # [1]
        y = torch.tensor(loc_seq[-1])

        # [1]
        return_dict["user"] = torch.tensor(selected["user_id"].values[0])
        # [sequence_len] binned in 15 minutes
        return_dict["time"] = torch.tensor(selected["start_min"].values[:-1] // 15 + 1)
        # [sequence_len]
        return_dict["weekday"] = torch.tensor(selected["weekday"].values[:-1] + 1, dtype=torch.int64)
        # [sequence_len] binned in 30 minutes
        return_dict["duration"] = torch.tensor(selected["act_duration"].values[:-1] // 30 + 1, dtype=torch.long)

        if self.if_embed_poi:
            return_dict["poi"] = torch.tensor(np.stack(selected["poiValues"].values[:-1]), dtype=torch.float32)

        # predict without padding, need to add padding in autoregressive prediction
        y_dict["duration"] = selected["act_duration"].values[-1] // 30
        return x, y, return_dict, y_dict


def collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch, tgt_batch = [], []

    # get one sample batch
    src_dict_batch = {"len": []}
    for key in batch[0][-2]:
        src_dict_batch[key] = []

    tgt_dict_batch = {}
    for key in batch[0][-1]:
        tgt_dict_batch[key] = []

    for src_sample, tgt_sample, src_dict, tgt_dict in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

        src_dict_batch["len"].append(len(src_sample))
        for key in src_dict:
            src_dict_batch[key].append(src_dict[key])

        for key in tgt_dict:
            tgt_dict_batch[key].append(tgt_dict[key])

    src_batch = pad_sequence(src_batch, padding_value=0)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)
    src_dict_batch["user"] = torch.tensor(src_dict_batch["user"], dtype=torch.int64)
    src_dict_batch["len"] = torch.tensor(src_dict_batch["len"], dtype=torch.int64)

    for key in src_dict_batch:
        if key in ["user", "len", "history_count"]:
            continue
        src_dict_batch[key] = pad_sequence(src_dict_batch[key], padding_value=0)
    for key in tgt_dict_batch:
        tgt_dict_batch[key] = torch.tensor(tgt_dict_batch[key], dtype=torch.float32)

    return src_batch, tgt_batch, src_dict_batch, tgt_dict_batch


def get_dataloaders(train_data, vali_data, test_data, config):
    # reindex the df for efficient selection
    train_data["id"] = np.arange(len(train_data))
    vali_data["id"] = np.arange(len(vali_data))
    test_data["id"] = np.arange(len(test_data))

    train_idx, vali_idx, test_idx = _get_valid_start_end_idx(
        train_data, vali_data, test_data, print_progress=config.verbose
    )

    dataset_train = traj_dataset(train_data, config, valid_start_end_idx=train_idx)
    dataset_val = traj_dataset(vali_data, config, valid_start_end_idx=vali_idx)
    dataset_test = traj_dataset(test_data, config, valid_start_end_idx=test_idx)

    kwds_train = {
        "shuffle": True,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": True,
    }
    kwds_val = {
        "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": True,
    }
    kwds_test = {
        "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": True,
    }

    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, **kwds_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=collate_fn, **kwds_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=collate_fn, **kwds_test)

    print(
        f"length of the train loader: {len(train_loader)}\t validation loader:{len(val_loader)}\t test loader:{len(test_loader)}"
    )
    return train_loader, val_loader, test_loader


def _get_valid_start_end_idx(train_data, vali_data, test_data, print_progress=True):
    train_idx = _get_valid_sequence(train_data, print_progress=print_progress)
    vali_idx = _get_valid_sequence(vali_data, print_progress=print_progress)
    test_idx = _get_valid_sequence(test_data, print_progress=print_progress)

    return train_idx, vali_idx, test_idx


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
        """
        Funtion warpper to parallelize funtions after .groupby().
        Parameters
        ----------
        dfGrouped: pd.DataFrameGroupBy
            The groupby object after calling df.groupby(COLUMN).
        func: function
            Function to apply to the dfGrouped object, i.e., dfGrouped.apply(func).
        n_jobs: int
            The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
            computing code is used at all, which is useful for debugging. See
            https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
            for a detailed description
        print_progress: boolean
            If set to True print the progress of apply.
        **kwargs:
            Other arguments passed to func.
        Returns
        -------
        pd.DataFrame:
            The result of dfGrouped.apply(func)
        """
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
