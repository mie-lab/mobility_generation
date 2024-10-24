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

from easydict import EasyDict as edict

from loc_predict.dataloader import load_data_predict
from loc_predict.utils import get_models, get_trained_nets, get_test_result, get_generated_sequences
from loc_predict.models.markov import markov_transition_prob
from utils.utils import load_data, setup_seed, load_config, init_save_path, get_train_test
from utils.dist_util import load_state_dict


def single_run(train_loader, val_loader, test_loader, config, device, log_dir):
    result_ls = []

    # get models
    model = get_models(config, device)

    # train, returns validation performances
    model, perf = get_trained_nets(config, model, train_loader, val_loader, device, log_dir)
    result_ls.append(perf)

    # test, return test performances
    perf = get_test_result(config, model, test_loader, device)

    result_ls.append(perf)

    return result_ls


def get_data_for_markov(type, level=13):
    sp = pd.read_csv(os.path.join(f"./data/sp_{type}.csv"), index_col="id")
    loc = pd.read_csv(os.path.join(f"./data/loc_s2_level10_{level}.csv"), index_col="id")

    sp = load_data(sp, loc)

    # get all possible locations
    all_locs = pd.read_csv(f"./data/s2_loc_visited_level10_{level}.csv", index_col="id")
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


if __name__ == "__main__":
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./config/markov.yml",
    )
    args = parser.parse_args()

    # initialization
    config = load_config(args.config)
    config = edict(config)

    setup_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_test = load_data_predict(batch_size=config.batch_size, data_args=config, split="test", shuffle=False)

    timestamp_now = int(datetime.datetime.now().timestamp())
    if "mhsa" in args.config:  # neural networks
        if not config.use_pretrain:  # for training
            data_train = load_data_predict(batch_size=config.batch_size, data_args=config, shuffle=True)
            data_valid = load_data_predict(batch_size=config.batch_size, data_args=config, split="valid", shuffle=False)

            # possibility to enable multiple runs
            result_ls = []
            for i in range(1):
                # train, validate and test
                log_dir = init_save_path(config, time_now=timestamp_now)
                # res_single contains the performance of validation and test of the current run
                res_single = single_run(data_train, data_valid, data_test, config, device, log_dir)
                result_ls.extend(res_single)

            # save results
            result_df = pd.DataFrame(result_ls)
            train_type = "default"
            filename = os.path.join(
                config.save_root, f"{config.dataset}_{config.networkName}_{train_type}_{str(timestamp_now)}.csv"
            )
            result_df.to_csv(filename, index=False)
        else:  # for generation
            model = get_models(config, device)
            checkpoint = load_state_dict(os.path.join(config.save_root, config.pretrain_filepath, config.model_name))
            model.load_state_dict(checkpoint["model"])

            generated_dict = get_generated_sequences(config, model, data_test, device)

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
    elif "markov" in args.config:  # markov model
        train_df, vali_df, test_df, all_locs_df = get_data_for_markov(type=config.dataset_variation, level=config.level)

        # construct markov matrix based on train and validation dataset
        train_vali_data = pd.concat([train_df, vali_df])

        transition_df = (
            train_vali_data.groupby(["user_id"]).apply(markov_transition_prob, n=config.n_dependence).reset_index()
        )

        groupby_transition = transition_df.groupby("user_id")

        generated_dict = get_generated_sequences(config, groupby_transition, data_test)

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

    else:
        raise AttributeError("Prediction method not implemented.")
