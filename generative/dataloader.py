import pickle as pickle

import datasets
import numpy as np
import psutil
import torch
import torch.distributed as dist
from datasets import Dataset as Dataset2
from datasets.utils.logging import disable_progress_bar
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

disable_progress_bar()

import trackintel as ti


class discriminator_dataset(torch.utils.data.Dataset):
    def __init__(self, true_data, fake_data):
        super(discriminator_dataset, self).__init__()

        # reindex the df for efficient selection
        self.true_data = true_data
        self.fake_data = fake_data

        self.true_len = len(true_data["locs"])
        self.fake_len = len(fake_data["locs"])

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.true_len + self.fake_len

    def __getitem__(self, idx):
        return_dict = {}

        if idx < self.true_len:
            loc_seq = self.true_data["locs"][idx]
            return_dict["duration"] = self.true_data["durs"][idx]
            y = 1
        else:
            fake_idx = idx - self.true_len

            # :-1 already constructed in construct_discriminator_pretrain_dataset()
            loc_seq = self.fake_data["locs"][fake_idx]
            return_dict["duration"] = self.fake_data["durs"][fake_idx]
            y = 0

        # [sequence_len]
        x = torch.tensor(loc_seq)
        return_dict["duration"] = torch.tensor(return_dict["duration"], dtype=torch.long)

        return x, y, return_dict


class generator_dataset(torch.utils.data.Dataset):
    def __init__(self, input_data):
        super(generator_dataset, self).__init__()

        self.data = input_data

        self.len = len(self.data["locs"])

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.len

    def __getitem__(self, idx):
        # [sequence_len]
        x = torch.tensor(self.data["locs"][idx][:-1])
        # [sequence_len]
        y = torch.tensor(self.data["locs"][idx][1:])

        return_dict = {}
        y_dict = {}
        #
        return_dict["duration"] = torch.tensor(self.data["durs"][idx][:-1], dtype=torch.long)
        # predict without padding, need to add padding in autoregressive prediction
        y_dict["duration"] = torch.tensor(self.data["durs"][idx][1:] - 1, dtype=torch.float32)

        return x, y, return_dict, y_dict


def discriminator_collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch, tgt_batch = [], []

    # get one sample batch
    src_dict_batch = {}
    for key in batch[0][-1]:
        src_dict_batch[key] = []

    for src_sample, tgt_sample, src_dict in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

        for key in src_dict:
            src_dict_batch[key].append(src_dict[key])

    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)

    for key in src_dict_batch:
        src_dict_batch[key] = pad_sequence(src_dict_batch[key], padding_value=0, batch_first=True)

    return src_batch, tgt_batch, src_dict_batch


def generator_collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch, tgt_batch = [], []

    # get one sample batch
    src_dict_batch = {}
    for key in batch[0][-2]:
        src_dict_batch[key] = []

    tgt_dict_batch = {}
    for key in batch[0][-1]:
        tgt_dict_batch[key] = []

    for src_sample, tgt_sample, src_dict, tgt_dict in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

        for key in src_dict:
            src_dict_batch[key].append(src_dict[key])

        for key in tgt_dict:
            tgt_dict_batch[key].append(tgt_dict[key])

    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0, batch_first=True)

    for key in src_dict_batch:
        src_dict_batch[key] = pad_sequence(src_dict_batch[key], padding_value=0, batch_first=True)
    for key in tgt_dict_batch:
        tgt_dict_batch[key] = pad_sequence(tgt_dict_batch[key], padding_value=0, batch_first=True)

    return src_batch, tgt_batch, src_dict_batch, tgt_dict_batch


def generate_samples(model, seq_len, num, single_len=256, print_progress=False):
    samples = {"locs": [], "durs": []}
    for _ in tqdm(range(int(num / single_len)), disable=not print_progress):
        gen_samples = model.module.sample(single_len, seq_len)

        samples["locs"].extend(gen_samples["locs"].detach().cpu().numpy().tolist())
        samples["durs"].extend(gen_samples["durs"].detach().cpu().numpy().tolist())

    samples["locs"] = np.array(samples["locs"])
    samples["durs"] = np.array(samples["durs"])
    return samples


def get_pretrain_loaders(input_data, world_size, config, device):
    train_data, vali_data = input_data

    # train dataset
    true_train, fake_train = _construct_discriminator_pretrain_dataset(train_data, config.max_location)
    true_vali, fake_vali = _construct_discriminator_pretrain_dataset(vali_data, config.max_location)

    d_train_data = discriminator_dataset(true_data=true_train, fake_data=fake_train)
    train_sampler = DistributedSampler(d_train_data, num_replicas=world_size, rank=device, shuffle=True, drop_last=True)
    d_train_loader = torch.utils.data.DataLoader(
        d_train_data,
        collate_fn=discriminator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        sampler=train_sampler,
    )

    # validation dataset
    d_vali_data = discriminator_dataset(true_data=true_vali, fake_data=fake_vali)
    vali_sampler = DistributedSampler(d_vali_data, num_replicas=world_size, rank=device, shuffle=False)
    d_vali_loader = torch.utils.data.DataLoader(
        d_vali_data,
        collate_fn=discriminator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        sampler=vali_sampler,
    )

    # training dataset
    g_train_data = generator_dataset(input_data=true_train)
    train_sampler = DistributedSampler(g_train_data, num_replicas=world_size, rank=device, shuffle=True)
    g_train_loader = torch.utils.data.DataLoader(
        g_train_data,
        collate_fn=generator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        sampler=train_sampler,
    )

    # validation dataset
    g_vali_data = generator_dataset(input_data=true_vali)
    vali_sampler = DistributedSampler(g_vali_data, num_replicas=world_size, rank=device, shuffle=False)
    g_vali_loader = torch.utils.data.DataLoader(
        g_vali_data,
        collate_fn=generator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        sampler=vali_sampler,
    )

    if is_main_process():
        print(f"len d_train loader:\t{len(d_train_loader)}\t #samples: {len(d_train_data)}")
        print(f"len d_vali loader:\t{len(d_vali_loader)}\t #samples: {len(d_vali_data)}")
        print(f"len g_train loader:\t{len(g_train_loader)}\t #samples: {len(g_train_data)}")
        print(f"len g_vali loader:\t{len(g_vali_loader)}\t #samples: {len(g_vali_data)}")

    return d_train_loader, d_vali_loader, g_train_loader, g_vali_loader


def _construct_discriminator_pretrain_dataset(input_data, max_location):
    true_seqs = {"locs": [], "durs": []}
    fake_seqs = {"locs": [], "durs": []}

    for x, _, x_dict, _ in tqdm(input_data):
        x = x.squeeze().numpy().copy()
        duration = x_dict["duration"].squeeze().numpy().copy()

        #
        loc_seq = x.copy()
        dur_seq = duration.copy()

        # random shuffle sequences; checked
        try:
            idx_arr = np.arange(len(loc_seq))
        except TypeError:
            print(loc_seq)
            continue
        np.random.shuffle(idx_arr)

        fake_seqs["locs"].append(loc_seq.copy()[idx_arr])
        fake_seqs["durs"].append(dur_seq.copy()[idx_arr])

        # random choose one location and switch to another location
        select_idx = np.random.randint(len(loc_seq), size=1)
        new_loc_seq = loc_seq.copy()
        new_dur_seq = dur_seq.copy()
        new_loc_seq[select_idx] = np.random.randint(low=2, high=max_location, size=1)
        new_dur_seq[select_idx] = np.random.randint(low=1, high=(60 * 24 * 2) // 30 + 1, size=1)

        fake_seqs["locs"].append(new_loc_seq)
        fake_seqs["durs"].append(new_dur_seq)

        true_seqs["locs"].append(x)
        true_seqs["durs"].append(duration)

    return true_seqs, fake_seqs


def get_discriminator_dataloaders(train_data, fake_data, world_size, config, device):
    d_train_data = discriminator_dataset(true_data=train_data, fake_data=fake_data)
    train_sampler = DistributedSampler(d_train_data, num_replicas=world_size, rank=device, shuffle=True)
    d_train_loader = torch.utils.data.DataLoader(
        d_train_data,
        collate_fn=discriminator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=True,
        sampler=train_sampler,
    )

    return d_train_data, d_train_loader


def load_data_diffusion(
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

    dataset = DiffSeqDataset(training_data, data_args, model_emb=model_emb)

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


def process_helper_fnc(seq_ls, split):
    seq_dataset = Dataset2.from_dict(seq_ls)

    def merge_and_mask(ls):
        MAX_LEN = 512
        GENERATE_LEN = 50

        src_ls = []
        src_xy_ls = []
        src_duration_ls = []

        tgt_ls = []
        tgt_duration_ls = []

        for i in range(len(ls["src"])):
            src = ls["src"][i]
            src_xy = ls["src_xy"][i]
            src_duration = ls["src_duration"][i]
            tgt = ls["tgt"][i]
            tgt_duration = ls["tgt_duration"][i]

            # for src
            len_src = len(src)
            if len_src > MAX_LEN:
                src = src[(len_src - MAX_LEN) :]
                src_xy = src_xy[(len_src - MAX_LEN) :]
                src_duration = src_duration[(len_src - MAX_LEN) :]

            # for tgt
            if split == "test":
                ori_len = len(tgt)
                if ori_len < GENERATE_LEN:  # pad with 0s to GENERATE_LEN
                    tgt = tgt + [0] * (GENERATE_LEN - ori_len)
                    tgt_duration = tgt_duration + [0] * (GENERATE_LEN - ori_len)
                else:
                    tgt = tgt[:GENERATE_LEN]
                    tgt_duration = tgt_duration[:GENERATE_LEN]
            else:
                if len(tgt) > MAX_LEN:
                    tgt = tgt[:MAX_LEN]
                    tgt_duration = tgt_duration[:MAX_LEN]

            src_ls.append(src)
            src_xy_ls.append(src_xy)
            src_duration_ls.append(src_duration)

            tgt_ls.append(tgt)
            tgt_duration_ls.append(tgt_duration)

        ls["tgt"] = tgt_ls
        ls["tgt_duration"] = tgt_duration_ls

        ls["src"] = src_ls
        ls["src_xy"] = src_xy_ls
        ls["src_duration"] = src_duration_ls
        return ls

    seq_dataset = seq_dataset.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc="merge and mask",
    )

    raw_datasets = datasets.DatasetDict()
    raw_datasets["train"] = seq_dataset

    return raw_datasets


def get_sequence(args, split="train"):
    print("#" * 30, "\nLoading dataset {} from {}...".format(args.dataset, args.data_dir))

    print(f"### Loading from the {split} set...")
    path = (
        f"{args.data_dir}/{split}_level{args.level}_{args.src_min_days}_{args.tgt_min_days}_{args.dataset_variation}.pk"
    )

    sequence_ls = pickle.load(open(path, "rb"))

    processed_dict = {"src": [], "src_xy": [], "src_duration": [], "tgt": [], "tgt_duration": []}
    for record in sequence_ls:
        processed_dict["src"].append(record["src"])
        processed_dict["src_xy"].append(record["src_xy"])

        # for padding, add normalization (max 2880 = 60 * 24 * 2 - 1 + 1 (padding))
        processed_dict["src_duration"].append((record["src_duration"] + 1) / 2880)

        processed_dict["tgt"].append(record["tgt"])

        processed_dict["tgt_duration"].append((record["tgt_duration"] + 1) / 2880)

    print("### Data samples...\n", processed_dict["src"][0][:5], processed_dict["tgt"][0][:5])

    train_dataset = process_helper_fnc(processed_dict, split)
    return train_dataset


class DiffSeqDataset(torch.utils.data.Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets["train"])
        self.data_args = data_args
        self.model_emb = model_emb

        self.if_embed_poi = data_args.if_embed_poi
        self.if_embed_xy = data_args.if_embed_xy

        self.if_include_duration = data_args.if_include_duration

        if self.if_embed_poi:
            poi_file_path = f"{data_args.data_dir}/poi_level{data_args.level}.npy"
            poi_file = np.load(poi_file_path, allow_pickle=True)
            self.poiValues = poi_file[()]["poiValues"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        current_data = self.text_datasets["train"][idx]

        src = torch.tensor(current_data["src"])
        tgt = torch.tensor(current_data["tgt"])

        src_ctx = {}
        tgt_cxt = {}
        if self.if_embed_xy:
            src_ctx["xy"] = torch.tensor(current_data["src_xy"], dtype=torch.float32)

        # construct the pois
        if self.if_embed_poi:
            ids = np.array(current_data["src"])
            pois = np.take(self.poiValues, ids - 1, axis=0)  # -1 for padding
            src_ctx["poi"] = torch.tensor(pois, dtype=torch.float32)

        if self.if_include_duration:
            src_ctx["duration"] = torch.tensor(current_data["src_duration"])
            tgt_cxt["duration"] = torch.tensor(current_data["tgt_duration"])

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
