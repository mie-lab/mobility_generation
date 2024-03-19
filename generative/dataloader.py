import numpy as np
import pickle as pickle
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import trackintel as ti


class discriminator_dataset(torch.utils.data.Dataset):
    def __init__(self, true_data, fake_data, valid_start_end_idx):
        super(discriminator_dataset, self).__init__()

        # reindex the df for efficient selection
        self.true_data = true_data.copy()
        self.true_data.set_index("id", inplace=True)

        self.fake_data = fake_data

        self.valid_start_end_idx = valid_start_end_idx
        self.true_len = len(valid_start_end_idx)
        self.fake_len = len(fake_data["locs"])

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.true_len + self.fake_len

    def __getitem__(self, idx):
        return_dict = {}

        if idx < self.true_len:
            start_end_idx = self.valid_start_end_idx[idx]
            selected = self.true_data.iloc[start_end_idx[0] : start_end_idx[1]]

            loc_seq = selected["location_id"].values[:-1]
            return_dict["duration"] = selected["act_duration"].values[:-1] // 30 + 1
            y = 1
        else:
            fake_idx = idx - self.true_len

            # :-1 already constructed in construct_discriminator_pretrain_dataset()
            loc_seq = self.fake_data["locs"][fake_idx]
            return_dict["duration"] = self.fake_data["durs"][fake_idx] // 30 + 1
            y = 0

        # [sequence_len]
        x = torch.tensor(loc_seq)
        return_dict["duration"] = torch.tensor(return_dict["duration"], dtype=torch.long)

        return x, y, return_dict


class generator_dataset(torch.utils.data.Dataset):
    def __init__(self, input_data, valid_start_end_idx):
        super(generator_dataset, self).__init__()

        self.data = input_data.copy()
        self.data.set_index("id", inplace=True)

        self.valid_start_end_idx = valid_start_end_idx
        self.len = len(valid_start_end_idx)

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.len

    def __getitem__(self, idx):
        start_end_idx = self.valid_start_end_idx[idx]
        selected = self.data.iloc[start_end_idx[0] : start_end_idx[1]]

        loc_seq = selected["location_id"].values
        # [sequence_len]
        x = torch.tensor(loc_seq[:-1])
        # [sequence_len]
        y = torch.tensor(loc_seq[1:])

        return_dict = {}
        y_dict = {}
        #
        return_dict["duration"] = torch.tensor(selected["act_duration"].values[:-1] // 30 + 1, dtype=torch.long)
        # predict without padding, need to add padding in autoregressive prediction
        y_dict["duration"] = torch.tensor(selected["act_duration"].values[1:] // 30, dtype=torch.float32)

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


def get_pretrain_loaders(input_data, all_locs, world_size, config, device):
    train_data, train_idx, vali_data, vali_idx = input_data

    # train dataset
    fake_train_samples = _construct_discriminator_pretrain_dataset(train_data, train_idx, all_locs)
    fake_vali_samples = _construct_discriminator_pretrain_dataset(vali_data, vali_idx, all_locs)

    d_train_data = discriminator_dataset(
        true_data=train_data, fake_data=fake_train_samples, valid_start_end_idx=train_idx
    )
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
    d_vali_data = discriminator_dataset(true_data=vali_data, fake_data=fake_vali_samples, valid_start_end_idx=vali_idx)
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
    g_train_data = generator_dataset(input_data=train_data, valid_start_end_idx=train_idx)
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
    g_vali_data = generator_dataset(input_data=vali_data, valid_start_end_idx=vali_idx)
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


def _construct_discriminator_pretrain_dataset(input_data, input_idx, all_locs):
    fake_seqs = {"locs": [], "durs": []}

    data_df = input_data.set_index("id")
    for start_idx, end_idx in tqdm(input_idx):
        curr_seq = data_df.iloc[start_idx:end_idx].copy()

        # only take the training idx
        loc_seq = curr_seq["location_id"].values[:-1]
        dur_seq = curr_seq["act_duration"].values[:-1]

        # random shuffle sequences; checked
        idx_arr = np.arange(len(loc_seq))
        np.random.shuffle(idx_arr)

        fake_seqs["locs"].append(loc_seq.copy()[idx_arr])
        fake_seqs["durs"].append(dur_seq.copy()[idx_arr])

        # random choose one location and switch to another location
        select_idx = np.random.randint(len(loc_seq), size=1)
        new_loc_seq = loc_seq.copy()
        new_dur_seq = dur_seq.copy()
        new_loc_seq[select_idx] = np.random.randint(low=1, high=len(all_locs) + 1, size=1)
        new_dur_seq[select_idx] = np.random.randint(low=0, high=60 * 24 * 2, size=1)

        fake_seqs["locs"].append(new_loc_seq)
        fake_seqs["durs"].append(new_dur_seq)

    return fake_seqs


def get_discriminator_dataloaders(train_data, train_idx, fake_data, world_size, config, device):
    d_train_data = discriminator_dataset(true_data=train_data, fake_data=fake_data, valid_start_end_idx=train_idx)
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
