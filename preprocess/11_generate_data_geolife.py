import numpy as np
import os
import pandas as pd
import geopandas as gpd
import pickle as pickle
import argparse


from shapely import wkt
from tqdm import tqdm

from joblib import Parallel, delayed

from sklearn.preprocessing import OrdinalEncoder

from utils.utils import _split_dataset


def save_pk_file(save_path, data):
    """Function to save data to pickle format given data and path."""
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def applyParallel(dfGrouped, func, n_jobs, print_progress=False, **kwargs):
    return Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )


def get_valid_data(df):
    valid_data = applyParallel(df.groupby("user_id"), _valid_sequence_user, n_jobs=-1)
    return [item for sublist in valid_data for item in sublist]


def get_train_test(sp, all_locs):
    sp = sp.copy()
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp.drop(columns={"started_at", "finished_at"}, inplace=True)
    sp["idx"] = sp.groupby("user_id").cumcount().add(1)

    # encoder user, 0 reserved for padding
    enc = OrdinalEncoder(dtype=np.int64)
    sp["user_id"] = enc.fit_transform(sp["user_id"].values.reshape(-1, 1)) + 1

    # encode mode
    enc = OrdinalEncoder(dtype=np.int64)
    sp["mode"] = enc.fit_transform(sp["mode"].values.reshape(-1, 1)) + 1
    # print(enc.categories_)

    # truncate too long duration, >2 days to 2 days
    sp.loc[sp["act_duration"] > 60 * 24 * 2, "act_duration"] = 60 * 24 * 2

    # split the datasets, user dependent 0.7, 0.2, 0.1
    train_data, vali_data, test_data = _split_dataset(sp)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        all_locs["id"].values.reshape(-1, 1)
    )
    # add 1 to account for 0 padding
    all_locs["id"] = enc.transform(all_locs["id"].values.reshape(-1, 1)) + 1
    #
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 1
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 1
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 1

    return train_data, vali_data, test_data, all_locs


def _valid_sequence_user(df):

    data_ls = []
    df = df.reset_index(drop=True).copy()

    min_days = df["start_day"].min()
    df["diff_day"] = df["start_day"] - min_days

    for index, row in df.iterrows():
        # exclude the first records
        if row["diff_day"] < SRC_MIN_DAYS:
            continue

        src_trace = df.iloc[: index + 1]
        src_trace = src_trace.loc[(src_trace["start_day"] >= (row["start_day"] - SRC_MAX_DAYS))]
        if len(src_trace) < 5:
            continue

        tgt_trace = df.iloc[index + 1 :]

        if ((tgt_trace["start_day"].max() - tgt_trace["start_day"].min()) < TGT_MIN_DAYS) or len(tgt_trace) < 2:
            continue
        if len(tgt_trace) > 256:
            tgt_trace = tgt_trace.head(256)

        curr_dict = {}
        curr_dict["src"] = src_trace["location_id"].values
        curr_dict["src_xy"] = src_trace[["x", "y"]].values
        curr_dict["src_duration"] = src_trace["act_duration"].values.astype(int)
        curr_dict["src_mode"] = src_trace["mode"].values
        curr_dict["src_user"] = src_trace["user_id"].values[0]
        curr_dict["src_weekday"] = src_trace["weekday"].values
        curr_dict["src_startmin"] = src_trace["start_min"].values
        # for mechanistic models
        curr_dict["src_idx"] = src_trace["idx"].values[-1]

        curr_dict["tgt"] = tgt_trace["location_id"].values
        curr_dict["tgt_xy"] = tgt_trace[["x", "y"]].values
        curr_dict["tgt_duration"] = tgt_trace["act_duration"].values.astype(int)
        curr_dict["tgt_mode"] = tgt_trace["mode"].values
        curr_dict["tgt_weekday"] = tgt_trace["weekday"].values
        curr_dict["tgt_startmin"] = tgt_trace["start_min"].values

        data_ls.append(curr_dict)

    return data_ls


def trim_data(ls, split="train"):
    return_ls = []
    for record in ls:
        len_src = len(record["src"])
        len_tgt = len(record["tgt"])

        for key, value in record.items():
            if key == "src_user" or key == "src_idx":
                continue

            # for every set: trim too long src
            if ("src" in key) and (len_src > MAX_LEN):
                record[key] = value[(len_src - MAX_LEN) :]

            if split == "train":  # for train and validation set
                if ("tgt" in key) and (len_tgt > MAX_LEN):
                    record[key] = value[:MAX_LEN]
            else:  # for test set
                if "tgt" in key:
                    if len_tgt < GENERATE_LEN:  # pad with 0s to GENERATE_LEN
                        record[key] = np.pad(value, (0, GENERATE_LEN - len_tgt), constant_values=0)
                    else:  # trim to GENERATE_LEN
                        record[key] = value[:GENERATE_LEN]
        return_ls.append(record)
    return return_ls


def load_data(sp, loc):
    sp = sp.merge(
        loc.reset_index().drop(columns={"user_id", "geometry"}), how="left", left_on="location_id", right_on="id"
    )
    sp = sp.drop(columns={"id", "extent"})
    sp.index.name = "id"
    sp.reset_index(inplace=True)

    sp["started_at"] = pd.to_datetime(sp["started_at"], format="mixed", yearfirst=True, utc=True).dt.tz_localize(None)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"], format="mixed", yearfirst=True, utc=True).dt.tz_localize(None)

    def _get_time_info(df):
        min_day = pd.to_datetime(df["started_at"].min().date())

        # get the alighned time with act_duration
        df["temp_time"] = pd.NA
        df["temp_time"] = df["finished_at"].shift(1)
        df.loc[df.index[0], "temp_time"] = df["started_at"].iloc[0]

        df["start_day"] = (df["temp_time"] - min_day).dt.days
        df["start_min"] = df["temp_time"].dt.hour * 60 + df["temp_time"].dt.minute
        df["weekday"] = df["temp_time"].dt.weekday

        df = df.drop(columns="temp_time")
        return df

    sp = sp.groupby("user_id").apply(_get_time_info, include_groups=False).reset_index().drop(columns={"level_1"})
    return sp


if __name__ == "__main__":
    MAX_LEN = 512
    GENERATE_LEN = 50

    SP_NAME = "sp_geolife_all"
    LOC_NAME = "loc_geolife"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_min_days", type=int, nargs="?", help="min days to consider for source sequence", default=7
    )
    parser.add_argument(
        "--src_max_days", type=int, nargs="?", help="max days to consider for source sequence", default=21
    )
    parser.add_argument(
        "--tgt_min_days", type=int, nargs="?", help="min days to consider for target sequence", default=3
    )
    args = parser.parse_args()

    SRC_MIN_DAYS = args.src_min_days
    SRC_MAX_DAYS = args.src_max_days
    TGT_MIN_DAYS = args.tgt_min_days

    sp = pd.read_csv(os.path.join(".", "data", f"{SP_NAME}.csv"), index_col="id")
    loc = pd.read_csv(os.path.join(".", "data", f"{LOC_NAME}.csv"), index_col="id")

    sp = load_data(sp, loc)
    sp["length"] = sp["length"] / 1000

    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)

    # geometry
    loc["geometry"] = loc["geometry"].apply(wkt.loads)
    loc = gpd.GeoDataFrame(loc, geometry="geometry", crs="EPSG:4326")
    # transform to projected coordinate systems
    loc = loc.to_crs("EPSG:4479")

    # coordinate information
    x = loc["geometry"].x.values
    y = loc["geometry"].y.values

    loc["x"] = (x - x.min()) / 1000
    loc["y"] = (y - y.min()) / 1000
    loc = loc.reset_index()

    # split data
    train_data, vali_data, test_data, all_locs = get_train_test(sp, all_locs=loc)

    # cleaning
    all_locs = all_locs.rename(columns={"id": "loc_id"})
    train_data = train_data.merge(all_locs[["loc_id", "x", "y"]], left_on="location_id", right_on="loc_id")
    vali_data = vali_data.merge(all_locs[["loc_id", "x", "y"]], left_on="location_id", right_on="loc_id")
    test_data = test_data.merge(all_locs[["loc_id", "x", "y"]], left_on="location_id", right_on="loc_id")

    train_data.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    vali_data.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    test_data.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)

    #
    valid_train_data = get_valid_data(train_data)
    valid_validation_data = get_valid_data(vali_data)
    valid_test_data = get_valid_data(test_data)

    valid_train_data = trim_data(valid_train_data)
    valid_validation_data = trim_data(valid_validation_data)
    valid_test_data = trim_data(valid_test_data, split="test")

    # sequences
    save_pk_file(os.path.join(".", "data", f"train_{SRC_MIN_DAYS}_{TGT_MIN_DAYS}_geolife.pk"), valid_train_data)

    save_pk_file(
        os.path.join(".", "data", f"valid_{SRC_MIN_DAYS}_{TGT_MIN_DAYS}_geolife.pk"),
        valid_validation_data,
    )

    save_pk_file(os.path.join(".", "data", f"test_{SRC_MIN_DAYS}_{TGT_MIN_DAYS}_geolife.pk"), valid_test_data)
