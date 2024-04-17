import argparse

import torch
import os

import pandas as pd
import geopandas as gpd
import datetime
import pickle
from shapely import wkt
import json
import numpy as np
from json import JSONEncoder

from tqdm import tqdm

from easydict import EasyDict as edict
from sklearn.preprocessing import OrdinalEncoder

from loc_predict.models.markov import markov_transition_prob
from utils.utils import load_data, setup_seed, load_config, _split_dataset
from utils.dist_util import load_state_dict
from sklearn.linear_model import LinearRegression
from mechanistic.dataloader import load_data_mechanistic
from mechanistic.models import EPR

from trackintel.geogr import point_haversine_dist
import powerlaw


def get_train_test(sp, all_locs=None):
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp.drop(columns={"started_at", "finished_at"}, inplace=True)
    sp["idx"] = sp.groupby("user_id").cumcount().add(1)

    # encoder user, 0 reserved for padding
    enc = OrdinalEncoder(dtype=np.int64)
    sp["user_id"] = enc.fit_transform(sp["user_id"].values.reshape(-1, 1)) + 1

    # truncate too long duration, >2 days to 2 days
    # sp.loc[sp["act_duration"] > 60 * 24 * 2 - 1, "act_duration"] = 60 * 24 * 2 - 1

    # split the datasets, user dependent 0.6, 0.2, 0.2
    train_data, vali_data, test_data = _split_dataset(sp)

    # encode unseen locations in validation and test into 0
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        all_locs["loc_id"].values.reshape(-1, 1)
    )
    # add 1 to account for 0 padding
    all_locs["loc_id"] = enc.transform(all_locs["loc_id"].values.reshape(-1, 1)) + 2

    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

    return train_data, vali_data, test_data, all_locs


def get_data_for_mechanistic(type):
    sp = pd.read_csv(os.path.join(f"./data/sp_{type}.csv"), index_col="id")
    loc = pd.read_csv(os.path.join("./data/loc_s2_level10_13.csv"), index_col="id")

    sp = load_data(sp, loc)

    # get all possible locations
    all_locs = pd.read_csv("./data/s2_loc_visited_level10_13.csv", index_col="id")
    all_locs["geometry"] = all_locs["geometry"].apply(wkt.loads)
    all_locs = gpd.GeoDataFrame(all_locs, geometry="geometry", crs="EPSG:4326")
    # transform to projected coordinate systems
    all_locs = all_locs.to_crs("EPSG:2056")

    train_data, vali_data, test_data, all_locs = get_train_test(sp, all_locs=all_locs)
    print(
        f"Max loc id {all_locs.loc_id.max()}, min loc id {all_locs.loc_id.min()}, unique loc id:{all_locs.loc_id.unique().shape[0]}"
    )

    return train_data, vali_data, test_data, all_locs


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def getAIC(fit, empr):
    aics = []

    aics.append(-2 * np.sum(fit.truncated_power_law.loglikelihoods(empr)) + 4)
    aics.append(-2 * np.sum(fit.power_law.loglikelihoods(empr)) + 2)
    aics.append(-2 * np.sum(fit.lognormal.loglikelihoods(empr)) + 4)

    aics = aics - np.min(aics)

    down = np.sum([np.exp(-aic / 2) for aic in aics])

    res = {}
    res["truncated_power_law"] = np.exp(-aics[0] / 2) / down
    res["power_law"] = np.exp(-aics[1] / 2) / down
    res["lognormal"] = np.exp(-aics[2] / 2) / down

    return res


def estimate_jump_length(df):
    # def get_jump_length(gdf):
    #     geom_arr = gdf.geometry.values

    #     res_ls = []
    #     for i in range(1, len(geom_arr)):
    #         res_ls.append(point_haversine_dist(geom_arr[i - 1].x, geom_arr[i - 1].y, geom_arr[i].x, geom_arr[i].y)[0])
    #     return res_ls

    # df["geometry"] = df["geometry"].apply(wkt.loads)
    # jump_length = gpd.GeoDataFrame(df, geometry="geometry").groupby("user_id").apply(get_jump_length)
    # flat_jump_length = np.array([item for sublist in jump_length.to_list() for item in sublist])
    # flat_jump_length = flat_jump_length[flat_jump_length > 20]

    # fit = powerlaw.Fit(flat_jump_length, xmin=20, xmin_distribution="lognormal")
    # print("AIC criteria for jump length:", getAIC(fit, flat_jump_length))
    # print(f"Lognormal: parameter1: {fit.lognormal.parameter1:.4f}\t parameter2: {fit.lognormal.parameter2:.4f}")
    # print(
    #     f"Truncated power law: parameter1: {fit.truncated_power_law.parameter1:.2f}\t parameter2: {fit.truncated_power_law.parameter2:.2f}"
    # )
    # print(f"Power law: alpha: {fit.power_law.alpha:.2f}")

    # return {"1": fit.lognormal.parameter1, "2": fit.lognormal.parameter2}
    return {"1": 7.4534, "2": 2.0797}


def estimate_wait_time(df):
    # duration_hour = df["act_duration"].values / 60
    # duration_hour = duration_hour[duration_hour > 0.1]

    # fit = powerlaw.Fit(duration_hour, xmin=0.1, xmin_distribution="lognormal")
    # print("AIC criteria for wait time:", getAIC(fit, duration_hour))
    # print(f"Lognormal: parameter1: {fit.lognormal.parameter1:.4f}\t parameter2: {fit.lognormal.parameter2:.4f}")
    # print(
    #     f"Truncated power law: parameter1: {fit.truncated_power_law.parameter1:.2f}\t parameter2: {fit.truncated_power_law.parameter2:.2f}"
    # )
    # print(f"Power law: alpha: {fit.power_law.alpha:.2f}")

    # return {"1": fit.lognormal.parameter1, "2": fit.lognormal.parameter2}
    return {"1": 0.9748, "2": 1.4316}


def get_parameter_estimate(df):
    df.sort_values(by=["start_day", "start_min"], inplace=True)

    loc = df["location_id"].values

    unique_count_ls = []
    for i in range(loc.shape[0]):
        unique_count_ls.append(len(np.unique(loc[: i + 1])))

    # big S
    unique_count_arr = np.array(unique_count_ls)

    # small n
    steps = np.arange(unique_count_arr.shape[0]) + 1

    logy = np.log(unique_count_arr)
    logx = np.log(steps)
    # print(logy, logx)
    reg = LinearRegression().fit(logx.reshape(-1, 1), logy)

    r = 1 / reg.coef_ - 1
    p = np.exp((reg.intercept_ - np.log(1 + r)) * (1 + r))

    return pd.Series([r[0], p[0]], index=["r", "p"])


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

    train_df, vali_df, test_df, all_locs_df = get_data_for_mechanistic(type=config.dataset_variation)

    # estimate user parameters based on train and validation dataset, checked!
    train_vali_data = pd.concat([train_df, vali_df])
    param_estimate = train_vali_data.groupby("user_id").apply(get_parameter_estimate).to_dict("index")
    param_jump_length = estimate_jump_length(train_vali_data)
    param_wait_time = estimate_wait_time(train_vali_data)

    # initialize mechanistic model
    model = EPR(all_locs_df, param_jump_length, param_wait_time)

    # test data sequences
    all_data = pd.concat([train_df, vali_df, test_df])
    data_test = load_data_mechanistic(batch_size=config.batch_size, data_args=config, split="test", shuffle=False)

    generated_dict = {"pred": []}
    for inputs in tqdm(data_test):
        x, _, x_dict, _ = inputs

        curr_user = x_dict["user"].numpy()[0]
        curr_idx = x_dict["idx"].numpy()[0]
        curr_param = param_estimate[curr_user]

        curr_train_seq = all_data.loc[all_data["user_id"] == curr_user, "location_id"].values[:curr_idx]

        gen_seq, gen_dur = model.simulate(
            train_seq=curr_train_seq, simulation_param=curr_param, length=config.generate_len
        )

        generated_dict["pred"].append(np.array(gen_seq).astype(int))

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
