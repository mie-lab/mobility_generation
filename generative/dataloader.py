import os
import numpy as np
import pickle as pickle
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
import torch

from joblib import Parallel, delayed
from pathlib import Path

from sklearn.preprocessing import OrdinalEncoder

import trackintel as ti


class discriminator_dataset(torch.utils.data.Dataset):
    def __init__(self, true_data, fake_data, valid_start_end_idx):
        super(discriminator_dataset, self).__init__()

        # reindex the df for efficient selection
        self.true_data = true_data
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

        self.data = input_data

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

        return x, y


def discriminator_collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch, tgt_batch = [], []

    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)

    return src_batch, tgt_batch


def generator_collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch, tgt_batch = [], []

    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch)
    tgt_batch = pad_sequence(tgt_batch)

    return src_batch, tgt_batch


def construct_discriminator_pretrain_dataset(config, train_data, train_idx, all_locs):
    save_path = os.path.join(config.temp_save_root, "temp", "discriminator_fake_dataset.pk")
    # if the file is pre-generated we load the file
    if Path(save_path).is_file():
        return pickle.load(open(save_path, "rb"))
    else:
        parent = Path(save_path).parent.absolute()
        if not os.path.exists(parent):
            os.makedirs(parent)

        fake_sequences = []
        for start_idx, end_idx in tqdm(train_idx):
            curr_seq = train_data.iloc[start_idx:end_idx]["location_id"].values

            random_seq = curr_seq.copy()
            np.random.shuffle(random_seq)
            fake_sequences.append(random_seq)

            # random choose one location and switch to another location
            selected_idx = np.random.randint(len(curr_seq), size=1)
            curr_seq[selected_idx] = np.random.randint(len(all_locs) + 1, size=1)

            fake_sequences.append(curr_seq)

        save_pk_file(save_path, fake_sequences)

        return fake_sequences


def save_pk_file(save_path, data):
    """Function to save data to pickle format given data and path."""
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
