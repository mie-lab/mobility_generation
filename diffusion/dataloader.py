import numpy as np
import pickle as pickle

import datasets
from datasets import Dataset as Dataset2
from datasets.utils.logging import disable_progress_bar

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler

disable_progress_bar()


def load_data(
    batch_size,
    shuffle=True,
    data_args=None,
    split="train",
    model_emb=None,
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

    training_data = get_sequence(data_args, split=split)

    dataset = MobilityDataset(training_data, data_args, model_emb=model_emb)

    if split == "test":
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=data_args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return data_loader


def process_helper_fnc(seq_ls):
    seq_dataset = Dataset2.from_dict(seq_ls)

    raw_datasets = datasets.DatasetDict()
    raw_datasets["train"] = seq_dataset

    return raw_datasets


def get_sequence(args, split="train"):
    print("#" * 30, "\nLoading dataset {} from {}...".format(args.dataset, args.data_dir))

    print(f"### Loading from the {split} set...")
    try:
        path = f"{args.data_dir}/{split}_level{args.level}_{args.src_min_days}_{args.tgt_min_days}_{args.dataset_variation}.pk"
    except AttributeError:
        path = f"{args.data_dir}/{split}_{args.src_min_days}_{args.tgt_min_days}_{args.dataset_variation}.pk"

    sequence_ls = pickle.load(open(path, "rb"))

    processed_dict = {
        "src": [],
        "src_xy": [],
        "src_duration": [],
        "src_time": [],
        "src_mode": [],
        "tgt": [],
        "tgt_time": [],
        "tgt_duration": [],
        "tgt_mode": [],
    }
    for record in sequence_ls:
        processed_dict["src"].append(record["src"])
        processed_dict["src_xy"].append(record["src_xy"])
        processed_dict["src_mode"].append(record["src_mode"])

        # time in minutes of day, add 1 for padding
        src_time = (2 * record["src_startmin"]) / 1440 - 1
        processed_dict["src_time"].append(src_time)

        # add normalization (max 2880 = 60 * 24 * 2), dur \in [0, 2880]
        src_dur = (2 * record["src_duration"]) / 2880 - 1
        processed_dict["src_duration"].append(src_dur)

        # time in minutes of day, add 1 for padding
        tgt_time = (2 * record["tgt_startmin"]) / 1440 - 1
        processed_dict["tgt_time"].append(tgt_time)

        processed_dict["tgt"].append(record["tgt"])
        processed_dict["tgt_mode"].append(record["tgt_mode"])

        # add normalization (max 2880 = 60 * 24 * 2), dur \in [-1, 1]
        tgt_dur = (2 * record["tgt_duration"]) / 2880 - 1
        processed_dict["tgt_duration"].append(tgt_dur)

    print("### Data samples...\n", processed_dict["src"][0][:5], processed_dict["tgt"][0][:5])

    train_dataset = process_helper_fnc(processed_dict)
    return train_dataset


class MobilityDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, data_args, model_emb=None):
        super().__init__()
        self.datasets = datasets
        self.length = len(self.datasets["train"])
        self.data_args = data_args
        self.model_emb = model_emb

        self.if_embed_context = data_args.if_embed_context

        self.if_include_duration = data_args.if_include_duration
        self.if_include_mode = data_args.if_include_mode

        if self.if_embed_context:
            poi_file_path = f"{data_args.data_dir}/poi_level{data_args.level}.npy"
            poi_file = np.load(poi_file_path, allow_pickle=True)
            self.poiValues = poi_file[()]["poiValues"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        current_data = self.datasets["train"][idx]

        src = torch.tensor(current_data["src"])
        tgt = torch.tensor(current_data["tgt"])

        src_ctx = {}
        tgt_cxt = {}
        if self.if_embed_context:
            src_ctx["xy"] = torch.tensor(current_data["src_xy"], dtype=torch.float32)

            ids = np.array(current_data["src"])
            pois = np.take(self.poiValues, ids - 1, axis=0)  # -1 for padding
            src_ctx["poi"] = torch.tensor(pois, dtype=torch.float32)

        if self.if_include_duration:
            src_ctx["duration"] = torch.tensor(current_data["src_duration"], dtype=torch.float32)
            tgt_cxt["duration"] = torch.tensor(current_data["tgt_duration"], dtype=torch.float32)

            src_ctx["time"] = torch.tensor(current_data["src_time"], dtype=torch.float32)
            tgt_cxt["time"] = torch.tensor(current_data["tgt_time"], dtype=torch.float32)

        if self.if_include_mode:
            src_ctx["mode"] = torch.tensor(current_data["src_mode"])
            tgt_cxt["mode"] = torch.tensor(current_data["tgt_mode"])

        return src, tgt, src_ctx, tgt_cxt


def collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch = []
    tgt_batch = []

    # get one sample batch
    src_ctx_batch = {}
    for key in batch[0][-2]:
        src_ctx_batch[key] = []
    tgt_ctx_batch = {}
    for key in batch[0][-1]:
        tgt_ctx_batch[key] = []

    for src, tgt, src_ctx, tgt_cxt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)

        for key in src_ctx:
            src_ctx_batch[key].append(src_ctx[key])
        for key in tgt_cxt:
            tgt_ctx_batch[key].append(tgt_cxt[key])

    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0, batch_first=True)

    for key in src_ctx_batch:
        if key in ["len"]:
            continue
        src_ctx_batch[key] = pad_sequence(src_ctx_batch[key], padding_value=0, batch_first=True)
    for key in tgt_ctx_batch:
        if key in ["len"]:
            continue
        tgt_ctx_batch[key] = pad_sequence(tgt_ctx_batch[key], padding_value=0, batch_first=True)

    return src_batch, tgt_batch, src_ctx_batch, tgt_ctx_batch


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
