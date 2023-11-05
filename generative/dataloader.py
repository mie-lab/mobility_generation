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


class NewGenIter(object):
    """Toy data iter to load digits"""

    def __init__(self, data_file, batch_size):
        super(NewGenIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = np.asarray(d, dtype="int64")
        data = torch.LongTensor(d[:, :-1])
        target = torch.LongTensor(d[:, 1:])
        self.idx += self.batch_size
        return data, target


def discriminator_collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch, tgt_batch = [], []

    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)

    return src_batch, tgt_batch
