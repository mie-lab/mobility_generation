import os
import numpy as np
import pickle as pickle
from tqdm import tqdm
from pandas.testing import assert_frame_equal

from torch.nn.utils.rnn import pad_sequence
import torch

from joblib import Parallel, delayed
from pathlib import Path

from sklearn.preprocessing import OrdinalEncoder

import trackintel as ti


class discriminator_dataset(torch.utils.data.Dataset):
    def __init__(self, true_data, fake_data, print_progress=True):
        # reindex the df for efficient selection
        true_data["id"] = np.arange(len(true_data))
        self.true_data = true_data
        self.fake_data = fake_data

        self.valid_start_end_idx = _get_valid_sequence(true_data, print_progress)
        self.true_len = len(self.valid_start_end_idx)
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
            loc_seq = self.fake_data[idx - self.true_len, :]
            y = 0

        # [sequence_len]
        x = torch.tensor(loc_seq)

        return x, y


def _get_valid_sequence(input_df, print_progress=True):
    def getValidSequenceUser(df, previous_day=7):
        id_ls = []
        df.reset_index(drop=True, inplace=True)

        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days

        for index, row in df.iterrows():
            # exclude the first records
            if row["diff_day"] < previous_day:
                continue

            curr_trace = df.iloc[: index + 1]
            curr_trace = curr_trace.loc[(curr_trace["start_day"] >= (row["start_day"] - previous_day))]

            # exclude series which contains too few records
            if len(curr_trace) > 2:
                id_ls.append([curr_trace["id"].values[0], curr_trace["id"].values[-1] + 1])

        return id_ls

    def applyParallel(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
        """
        Funtion warpper to parallelize funtions after .groupby().
        Parameters
        ----------
        dfGrouped: pd.DataFrameGroupBy
            The groupby object after calling df.groupby(COLUMN).
        func: function
            Function to apply to the dfGrouped object, i.e., dfGrouped.apply(func).
        n_jobs: int
            The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
            computing code is used at all, which is useful for debugging. See
            https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
            for a detailed description
        print_progress: boolean
            If set to True print the progress of apply.
        **kwargs:
            Other arguments passed to func.
        Returns
        -------
        pd.DataFrame:
            The result of dfGrouped.apply(func)
        """
        return Parallel(n_jobs=n_jobs)(
            delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
        )

    valid_user_ls = applyParallel(
        input_df.groupby("user_id"), getValidSequenceUser, n_jobs=-1, previous_day=7, print_progress=print_progress
    )
    return [item for sublist in valid_user_ls for item in sublist]
