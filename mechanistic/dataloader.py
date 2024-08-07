import pickle as pickle

import datasets
import numpy as np
import torch
import trackintel as ti
from datasets import Dataset as Dataset2
from datasets.utils.logging import disable_progress_bar

from torch.nn.utils.rnn import pad_sequence

disable_progress_bar()


def load_data_mech(
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

    dataset = MechanisticDataset(data_sequence)

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

    processed_dict = {"src": [], "user": [], "idx": [], "tgt": [], "tgt_duration": []}
    for record in sequence_ls:
        processed_dict["src"].append(record["src"])
        processed_dict["user"].append(record["src_user"])
        processed_dict["idx"].append(record["src_idx"])

        processed_dict["tgt"].append(record["tgt"][0])
        processed_dict["tgt_duration"].append(record["tgt_duration"][0] // 30)

    print(
        "### Data samples...\n",
        processed_dict["src"][:1],
        processed_dict["tgt"][:1],
        processed_dict["tgt_duration"][:1],
    )
    raw_datasets = datasets.DatasetDict()
    raw_datasets["data"] = Dataset2.from_dict(processed_dict)

    return raw_datasets


class MechanisticDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets["data"]
        self.length = len(self.datasets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        selected = self.datasets[idx]
        x = torch.tensor(selected["src"])
        y = torch.tensor(selected["tgt"])

        x_dict = {}
        y_dict = {}

        x_dict["user"] = torch.tensor(selected["user"])
        x_dict["idx"] = torch.tensor(selected["idx"])

        y_dict["duration"] = selected["tgt_duration"]

        return x, y, x_dict, y_dict


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

    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)
    src_dict_batch["user"] = torch.tensor(src_dict_batch["user"], dtype=torch.int64)
    src_dict_batch["idx"] = torch.tensor(src_dict_batch["idx"], dtype=torch.int64)
    src_dict_batch["len"] = torch.tensor(src_dict_batch["len"], dtype=torch.int64)

    for key in src_dict_batch:
        if key in ["user", "len", "idx", "history_count"]:
            continue
        src_dict_batch[key] = pad_sequence(src_dict_batch[key], padding_value=0, batch_first=True)
    for key in tgt_dict_batch:
        tgt_dict_batch[key] = torch.tensor(tgt_dict_batch[key], dtype=torch.float32)

    return src_batch, tgt_batch, src_dict_batch, tgt_dict_batch
