import os
import numpy as np
import pickle as pickle
from tqdm import tqdm

from joblib import Parallel, delayed
from pathlib import Path

from sklearn.preprocessing import OrdinalEncoder


def prepare_nn_dataset(source_file, temp_save_root):
    train_data, vali_data, test_data = _split_train_test(source_file)

    print(
        f"Max location id:{train_data.location_id.max()}, unique location id:{train_data.location_id.unique().shape[0]}"
    )

    _generate_temp_datasets(train_data, temp_save_root, "train")
    _generate_temp_datasets(vali_data, temp_save_root, "validation")
    _generate_temp_datasets(test_data, temp_save_root, "test")

    return train_data.location_id.max(), train_data.user_id.max()


def _split_train_test(source_file):
    source_file.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)

    # encoder user, 0 reserved for padding
    enc = OrdinalEncoder(dtype=np.int64)
    source_file["user_id"] = enc.fit_transform(source_file["user_id"].values.reshape(-1, 1)) + 1

    # truncate too long duration, >2 days to 2 days
    source_file.loc[source_file["duration"] > 60 * 24 * 2 - 1, "duration"] = 60 * 24 * 2 - 1

    # split the datasets, user dependent 0.6, 0.2, 0.2
    train_data, vali_data, test_data = _split_dataset(source_file)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(
        dtype=np.int64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ).fit(train_data["location_id"].values.reshape(-1, 1))
    # add 2 to account for unseen locations (1) and to account for 0 padding
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

    return train_data, vali_data, test_data


def _generate_temp_datasets(data, temp_save_root, dataset_type):
    """Generate the datasets and save to the disk."""
    save_path = os.path.join(temp_save_root, "temp", f"predict_{dataset_type}.pk")
    if not Path(save_path).is_file():
        valid_records = _get_valid_sequence(data)

        # saving
        parent_dir = Path(save_path).parent.absolute()
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        with open(save_path, "wb") as handle:
            pickle.dump(valid_records, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _split_dataset(totalData):
    """Split dataset into train, vali and test."""

    def getSplitDaysUser(df):
        """Split the dataset according to the tracked day of each user."""
        maxDay = df["start_day"].max()
        train_split = maxDay * 0.6
        vali_split = maxDay * 0.8

        df["Dataset"] = "test"
        df.loc[df["start_day"] < train_split, "Dataset"] = "train"
        df.loc[
            (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
            "Dataset",
        ] = "vali"

        return df

    totalData = totalData.groupby("user_id", group_keys=False).apply(getSplitDaysUser)

    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # final cleaning
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data


def _get_valid_sequence(input_df):
    def getValidSequenceUser(df, previous_day=7):
        df.reset_index(drop=True, inplace=True)

        data_single_user = []
        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days

        for index, row in df.iterrows():
            # exclude the first records
            if row["diff_day"] < previous_day:
                continue

            hist = df.iloc[:index]
            hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]

            # exclude series which contains too few records
            if len(hist) < 3:
                continue

            data_dict = {}
            data_dict["X"] = hist["location_id"].values
            data_dict["user_X"] = hist["user_id"].values
            data_dict["start_min_X"] = hist["start_min"].values
            data_dict["weekday_X"] = hist["weekday"].values
            # data_dict["end_min_X"] = curr["end_min"].values
            data_dict["duration_X"] = hist["duration"].values

            # the next location is the target
            data_dict["Y"] = int(row["location_id"])

            data_single_user.append(data_dict)

        return data_single_user

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

    valid_user_ls = applyParallel(input_df.groupby("user_id"), getValidSequenceUser, n_jobs=-1)
    return [item for sublist in valid_user_ls for item in sublist]
