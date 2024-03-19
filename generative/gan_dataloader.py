import os
import numpy as np
import pickle as pickle
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
import torch


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
        self.fake_len = len(fake_data)

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.true_len + self.fake_len

    def __getitem__(self, idx):
        if idx < self.true_len:
            start_end_idx = self.valid_start_end_idx[idx]
            selected = self.true_data.iloc[start_end_idx[0] : start_end_idx[1]]
            loc_seq = selected["location_id"].values[:-1]
            y = 1
        else:
            loc_seq = self.fake_data[idx - self.true_len]
            y = 0

        # [sequence_len]
        x = torch.tensor(loc_seq)

        return x, y


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

    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)

    return src_batch, tgt_batch


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


def construct_discriminator_pretrain_dataset(input_data, input_idx, all_locs):
    fake_sequences = []

    data_df = input_data.set_index("id")
    for start_idx, end_idx in input_idx:
        curr_seq = data_df.iloc[start_idx:end_idx]["location_id"].values.copy()

        random_seq = curr_seq.copy()
        np.random.shuffle(random_seq)
        fake_sequences.append(random_seq)

        # random choose one location and switch to another location
        selected_idx = np.random.randint(len(curr_seq), size=1)
        curr_seq[selected_idx] = np.random.randint(low=1, high=len(all_locs) + 1, size=1)

        fake_sequences.append(curr_seq)

    return fake_sequences


def save_pk_file(save_path, data):
    """Function to save data to pickle format given data and path."""
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
