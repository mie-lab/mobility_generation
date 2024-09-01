import argparse

import os

import pandas as pd
import geopandas as gpd
import datetime
from shapely import wkt
import json
import numpy as np
from json import JSONEncoder

from tqdm import tqdm

from easydict import EasyDict as edict
from sklearn.preprocessing import OrdinalEncoder

from utils.utils import load_data, setup_seed, load_config, _split_dataset
from mechanistic.dataloader import load_data_mech
from mechanistic.container import ScalesOptim, recover_parameters_from_fitted_trace, generate_trace


def get_train_test_mech(sp, all_locs=None):
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp.drop(columns={"started_at", "finished_at"}, inplace=True)
    sp["idx"] = sp.groupby("user_id").cumcount().add(1)

    # encoder user, 0 reserved for padding
    enc = OrdinalEncoder(dtype=np.int64)
    sp["user_id"] = enc.fit_transform(sp["user_id"].values.reshape(-1, 1)) + 1

    # split the datasets, user dependent 0.7, 0.2, 0.1
    train_data, vali_data, test_data = _split_dataset(sp)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        all_locs["loc_id"].values.reshape(-1, 1)
    )
    # add 1 to account for 0 padding
    all_locs["loc_id"] = enc.transform(all_locs["loc_id"].values.reshape(-1, 1)) + 1

    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 1
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 1
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 1

    return train_data, vali_data, test_data, all_locs


def get_data_mech(type):
    sp = pd.read_csv(os.path.join(f"./data/sp_{type}.csv"), index_col="id")
    loc = pd.read_csv(os.path.join("./data/loc_s2_level10_14.csv"), index_col="id")

    sp = load_data(sp, loc)

    # get all possible locations
    all_locs = pd.read_csv("./data/s2_loc_visited_level10_14.csv", index_col="id")
    all_locs["geometry"] = all_locs["geometry"].apply(wkt.loads)
    all_locs = gpd.GeoDataFrame(all_locs, geometry="geometry", crs="EPSG:4326")
    # transform to projected coordinate systems
    # all_locs = all_locs.to_crs("EPSG:2056")

    train_data, vali_data, test_data, all_locs = get_train_test_mech(sp, all_locs=all_locs)
    print(
        f"Max loc id {all_locs.loc_id.max()}, min loc id {all_locs.loc_id.min()}, unique loc id:{all_locs.loc_id.unique().shape[0]}"
    )

    return train_data, vali_data, test_data, all_locs


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


if __name__ == "__main__":
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./config/mechanistic.yml",
    )
    args = parser.parse_args()

    # initialization
    config = load_config(args.config)
    config = edict(config)

    setup_seed(config.seed)
    timestamp_now = int(datetime.datetime.now().timestamp())

    train_df, vali_df, test_df, all_locs_df = get_data_mech(type=config.dataset_variation)

    df = pd.concat([train_df, vali_df]).sort_values(by=["user_id", "start_day", "start_min"])

    # fit
    result_dict = {}
    for curr_user in tqdm(df["user_id"].unique()):
        result_dict[curr_user] = {}
        curr_df = df.loc[df["user_id"] == curr_user].copy()

        # stop_locations
        user_unq_locs = np.sort(curr_df["location_id"].unique())
        locs_geom = all_locs_df.loc[all_locs_df["loc_id"].isin(user_unq_locs)]
        lat_lng = np.stack([locs_geom.geometry.y.values, locs_geom.geometry.x.values]).T

        # labels
        labels = curr_df["location_id"].map(dict(zip(user_unq_locs, range(len(user_unq_locs))))).values

        final_series, _, likelihoods, _, final_sizes, _, _ = ScalesOptim(
            labels=labels, stop_locations=lat_lng, bootstrap=False, nprocs=1, min_dist=1.2, verbose=False
        ).find_best_scale()

        # save the location mapping
        result_dict[curr_user]["user_locs"] = user_unq_locs
        result_dict[curr_user]["final_series"] = final_series

        # print(final_series)
        # break

    # generate
    data_test = load_data_mech(batch_size=config.batch_size, data_args=config, split="test", shuffle=False)

    generated_dict = {"pred": []}
    for inputs in tqdm(data_test):
        x, _, x_dict, _ = inputs

        curr_user = x_dict["user"].numpy()[0]
        curr_param = result_dict[curr_user]

        # generate
        final_series = curr_param["final_series"]
        nested_dictionary, cell_p_change = recover_parameters_from_fitted_trace(final_series)
        new_trace = generate_trace(
            nested_dictionary, cell_p_change, config.generate_len + 1, initial_position=final_series[-1]
        )

        # transform to original loc_id via location mapping
        loc_mapped = np.take(curr_param["user_locs"], np.array(new_trace)).astype(int)

        generated_dict["pred"].append(loc_mapped[1:])

    filename = os.path.join(
        config.save_root, f"{config.dataset}_{config.networkName}_generation_{str(timestamp_now)}.json"
    )
    fout = open(filename, "a")
    for recov in generated_dict["pred"]:
        print(
            json.dumps({"recover": recov}, cls=NumpyArrayEncoder),
            file=fout,
        )
    fout.close()
