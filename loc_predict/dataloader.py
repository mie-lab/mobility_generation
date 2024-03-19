import numpy as np
import pickle as pickle


from torch.nn.utils.rnn import pad_sequence
import torch


import trackintel as ti

from utils.utils import get_valid_start_end_idx


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

    train_idx, vali_idx, test_idx = get_valid_start_end_idx(
        train_data=train_data, vali_data=vali_data, test_data=test_data, print_progress=config.verbose, return_test=True
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
