import pickle as pickle

import datasets
import numpy as np
import torch
import trackintel as ti
from datasets import Dataset as Dataset2
from datasets.utils.logging import disable_progress_bar

from torch.nn.utils.rnn import pad_sequence

from utils.utils import get_valid_start_end_idx

disable_progress_bar()


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
        # TODO: check with embedding, regression or classification?
        y_dict["duration"] = selected["act_duration"].values[-1] // 30
        return x, y, return_dict, y_dict


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


def load_data_predict(
    batch_size,
    shuffle=True,
    data_args=None,
    split="train",
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print("#" * 30, "\nLoading location data...")

    data_sequence = get_sequence(data_args, split=split)

    dataset = PredictionDataset(data_sequence, data_args)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn
    )

    return data_loader


def get_sequence(args, split="train"):
    print("#" * 30, "\nLoading dataset {} from {}...".format(args.dataset, args.data_dir))

    print(f"### Loading from the {split} set...")
    path = (
        f"{args.data_dir}/{split}_level{args.level}_{args.src_min_days}_{args.tgt_min_days}_{args.dataset_variation}.pk"
    )

    sequence_ls = pickle.load(open(path, "rb"))

    processed_dict = {
        "src": [],
        "user": [],
        "src_time": [],
        "src_duration": [],
        "src_weekday": [],
        # "src_mode": [],
        "tgt": [],
        "tgt_duration": [],
        # "tgt_mode": [],
    }

    for record in sequence_ls:
        processed_dict["src"].append(record["src"])
        processed_dict["tgt"].append(record["tgt"][0])

        # Markov and mhsa
        processed_dict["user"].append(record["src_user"])

        # from original mhsa paper
        # binned into 15 min, add 1 for padding
        processed_dict["src_time"].append(record["src_startmin"] // 15 + 1)
        # add 1 for padding
        processed_dict["src_weekday"].append(record["src_weekday"] + 1)

        # attributes as input
        # processed_dict["src_mode"].append(record["src_mode"])
        # for padding, add normalization to [-1, 1] (max 2880 = 60 * 24 * 2 - 1 + 1 (padding))
        # dur = (2 * (record["src_duration"] + 1) / 2880) - 1
        # processed_dict["src_duration"].append(dur)

        # for GAN
        src_duration = record["src_duration"]
        src_duration[src_duration == 2880] = 2879
        processed_dict["src_duration"].append(src_duration // 30 + 1)

        # attributes as output
        # dur = (2 * (record["tgt_duration"][0] + 1) / 2880) - 1
        # processed_dict["tgt_duration"].append(dur)

        # for GAN
        tgt_duration = record["tgt_duration"]
        tgt_duration[tgt_duration == 2880] = 2879
        processed_dict["tgt_duration"].append(tgt_duration[0] // 30 + 1)

        # processed_dict["tgt_mode"].append(record["tgt_mode"][0])

    print("### Data samples...\n", processed_dict["src"][0][:5], processed_dict["tgt"][0])

    raw_datasets = datasets.DatasetDict()
    raw_datasets["data"] = Dataset2.from_dict(processed_dict)

    return raw_datasets


class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, data_args):
        super().__init__()
        self.datasets = datasets
        self.length = len(self.datasets["data"])

        self.if_embed_time = data_args.if_embed_time
        self.if_embed_poi = data_args.if_embed_poi

        self.if_include_duration = data_args.if_include_duration
        self.if_include_mode = data_args.if_include_mode

        if self.if_embed_poi:
            poi_file_path = f"{data_args.data_dir}/poi_level{data_args.level}.npy"
            poi_file = np.load(poi_file_path, allow_pickle=True)
            self.poiValues = poi_file[()]["poiValues"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        selected = self.datasets["data"][idx]
        src = torch.tensor(selected["src"])
        tgt = torch.tensor(selected["tgt"])

        src_ctx = {}
        tgt_cxt = {}

        src_ctx["user"] = torch.tensor(selected["user"], dtype=torch.int64)

        if self.if_embed_time:
            src_ctx["time"] = torch.tensor(selected["src_time"], dtype=torch.int64)
            src_ctx["weekday"] = torch.tensor(selected["src_weekday"], dtype=torch.int64)

        # construct the pois
        if self.if_embed_poi:
            ids = np.array(selected["src"])
            pois = np.take(self.poiValues, ids - 1, axis=0)  # -1 for padding
            src_ctx["poi"] = torch.tensor(pois, dtype=torch.float32)

        if self.if_include_duration:
            src_ctx["duration"] = torch.tensor(selected["src_duration"])
            tgt_cxt["duration"] = torch.tensor(selected["tgt_duration"])

        if self.if_include_mode:
            src_ctx["mode"] = torch.tensor(selected["src_mode"])
            tgt_cxt["mode"] = torch.tensor(selected["tgt_mode"])

        return src, tgt, src_ctx, tgt_cxt


def collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch = []
    tgt_batch = []

    # get one sample batch
    src_dict_batch = {"len": []}
    for key in batch[0][-2]:
        src_dict_batch[key] = []

    tgt_dict_batch = {}
    for key in batch[0][-1]:
        tgt_dict_batch[key] = []

    for src, tgt, src_dict, tgt_dict in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)

        src_dict_batch["len"].append(len(src))
        for key in src_dict:
            src_dict_batch[key].append(src_dict[key])

        for key in tgt_dict:
            tgt_dict_batch[key].append(tgt_dict[key])

    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)
    src_dict_batch["user"] = torch.tensor(src_dict_batch["user"], dtype=torch.int64)
    src_dict_batch["len"] = torch.tensor(src_dict_batch["len"], dtype=torch.int64)

    for key in src_dict_batch:
        if key in ["user", "len", "history_count"]:
            continue
        src_dict_batch[key] = pad_sequence(src_dict_batch[key], padding_value=0, batch_first=True)
    for key in tgt_dict_batch:
        tgt_dict_batch[key] = torch.tensor(tgt_dict_batch[key], dtype=torch.float32)

    return src_batch, tgt_batch, src_dict_batch, tgt_dict_batch
