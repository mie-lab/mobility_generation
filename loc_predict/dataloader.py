import os

from torch.nn.utils.rnn import pad_sequence
import torch

import pickle as pickle
import trackintel as ti


class traj_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data = pickle.load(open(data_dir, "rb"))

        self.len = len(self.data)

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.len

    def __getitem__(self, idx):
        selected = self.data[idx]

        return_dict = {}
        # [sequence_len]
        x = torch.tensor(selected["X"])
        # [1]
        y = torch.tensor(selected["Y"])

        # [1]
        return_dict["user"] = torch.tensor(selected["user_X"][0])
        # [sequence_len] in 15 minutes
        return_dict["time"] = torch.tensor(selected["start_min_X"] // 15)
        # [sequence_len]
        return_dict["weekday"] = torch.tensor(selected["weekday_X"], dtype=torch.int64)
        return x, y, return_dict


def collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch, tgt_batch = [], []

    # get one sample batch
    dict_batch = {"len": []}
    for key in batch[0][-1]:
        dict_batch[key] = []

    for src_sample, tgt_sample, return_dict in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

        dict_batch["len"].append(len(src_sample))

        for key in return_dict:
            dict_batch[key].append(return_dict[key])

    src_batch = pad_sequence(src_batch)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)
    dict_batch["user"] = torch.tensor(dict_batch["user"], dtype=torch.int64)
    dict_batch["len"] = torch.tensor(dict_batch["len"], dtype=torch.int64)

    for key in dict_batch:
        if key in ["user", "len", "history_count"]:
            continue
        dict_batch[key] = pad_sequence(dict_batch[key])

    return src_batch, tgt_batch, dict_batch


def get_dataloaders(config):
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

    dataset_train = traj_dataset(os.path.join(config.temp_save_root, "temp", "train.pk"))
    dataset_val = traj_dataset(os.path.join(config.temp_save_root, "temp", "validation.pk"))
    dataset_test = traj_dataset(os.path.join(config.temp_save_root, "temp", "test.pk"))

    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, **kwds_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=collate_fn, **kwds_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=collate_fn, **kwds_test)

    print(
        f"length of the train loader: {len(train_loader)}\t validation loader:{len(val_loader)}\t test loader:{len(test_loader)}"
    )
    return train_loader, val_loader, test_loader
