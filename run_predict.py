import argparse

import torch
import os

import pandas as pd
import datetime
import pickle

from easydict import EasyDict as edict

from utils.dataloader import get_dataloaders, get_train_test
from loc_predict.utils import get_models, get_trained_nets, get_test_result, get_generated_sequences
from loc_predict.models.markov import markov_transition_prob
from utils.utils import load_data, setup_seed, load_config, init_save_path


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


if __name__ == "__main__":
    setup_seed(1)

    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./config/mhsa.yml",
    )
    args = parser.parse_args()

    # initialization
    config = load_config(args.config)
    config = edict(config)

    # read and preprocess
    sp = pd.read_csv(os.path.join(config.temp_save_root, "sp_small.csv"), index_col="id")
    loc = pd.read_csv(os.path.join(config.temp_save_root, "loc_s2_level10_14.csv"))
    if config.if_embed_poi:
        loc_poi = pd.DataFrame(pickle.load(open(os.path.join(config.temp_save_root, "s2_loc_poi_level10_14.pk"), "rb")))
        loc = loc.merge(loc_poi, left_on="s2_id", right_on="loc_id", how="left").drop(columns={"loc_id"})
    sp = load_data(sp, loc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # get dataloaders
    train_data, vali_data, test_data = get_train_test(sp)
    print(
        f"Max location id:{train_data.location_id.max()}, unique location id:{train_data.location_id.unique().shape[0]}"
    )
    config["total_loc_num"] = int(train_data.location_id.max() + 1)
    config["total_user_num"] = int(train_data.user_id.max() + 1)

    train_loader, val_loader, test_loader = get_dataloaders(train_data, vali_data, test_data, config)

    if "mhsa" in args.config:  # neural networks
        if not config.use_pretrain:  # for training
            # possibility to enable multiple runs
            result_ls = []
            for i in range(2):
                # train, validate and test
                log_dir = init_save_path(config, time_now=int(datetime.datetime.now().timestamp()))
                # res_single contains the performance of validation and test of the current run
                res_single = single_run(train_loader, val_loader, test_loader, config, device, log_dir)
                result_ls.extend(res_single)

            # save results
            result_df = pd.DataFrame(result_ls)
            train_type = "all"
            filename = os.path.join(
                config.save_root,
                f"{config.dataset}_{config.networkName}_{train_type}_{str(int(datetime.datetime.now().timestamp()))}.csv",
            )
            result_df.to_csv(filename, index=False)
        else:  # for generation
            model = get_models(config, device)
            model.load_state_dict(torch.load(os.path.join(config.save_root, config.pretrain_filepath, "checkpoint.pt")))

            generated_df = get_generated_sequences(config, model, test_loader, device)

            filename = os.path.join(
                config.save_root,
                f"{config.dataset}_{config.networkName}_generation_{str(int(datetime.datetime.now().timestamp()))}.csv",
            )
            generated_df.to_csv(filename, index=True)

    elif "markov" in args.config:  # markov model
        train_data, vali_data, test_data = get_train_test(sp)

        # construct markov matrix based on train and validation dataset
        train_vali_data = pd.concat([train_data, vali_data])

        transition_df = (
            train_vali_data.groupby(["user_id"]).apply(markov_transition_prob, n=config.n_dependence).reset_index()
        )

        groupby_transition = transition_df.groupby("user_id")

        generated_df = get_generated_sequences(config, groupby_transition, test_loader)
        filename = os.path.join(
            config.save_root,
            f"{config.dataset}_{config.networkName}_generation_{str(int(datetime.datetime.now().timestamp()))}.csv",
        )
        generated_df.to_csv(filename, index=True)

    else:
        raise AttributeError("Prediction method not implemented.")
