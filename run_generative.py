# coding=utf-8
import os
import torch
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd


from shapely import wkt


from easydict import EasyDict as edict


from utils.utils import load_data, setup_seed, load_config, init_save_path
from utils.dataloader import get_train_test, _get_valid_sequence

from generative.movesim import Discriminator, Generator, AllEmbedding
from generative.train import pre_training, adv_training


if __name__ == "__main__":
    setup_seed(0)

    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Path to the config file.",
        default="./config/movesim.yml",
    )
    args = parser.parse_args()

    # initialization
    config = load_config(args.config)
    config = edict(config)

    # read and preprocess
    sp = pd.read_csv(os.path.join(config.temp_save_root, "sp.csv"), index_col="id")
    loc = pd.read_csv(os.path.join(config.temp_save_root, "locs_s2.csv"), index_col="id")
    sp = load_data(sp, loc)

    all_locs = pd.read_csv(os.path.join(config.temp_save_root, "all_locations.csv"), index_col="id")
    all_locs["geometry"] = all_locs["geometry"].apply(wkt.loads)
    all_locs = gpd.GeoDataFrame(all_locs, geometry="geometry", crs="EPSG:4326")
    # transform to projected coordinate systems
    all_locs = all_locs.to_crs("EPSG:2056")

    train_data, vali_data, test_data, all_locs = get_train_test(sp, all_locs=all_locs)

    print(f"Max location id:{all_locs.loc_id.max()}, unique location id:{all_locs.loc_id.unique().shape[0]}")
    config["total_loc_num"] = int(all_locs.loc_id.max() + 1)
    config["total_user_num"] = int(train_data.user_id.max() + 1)

    # get valid idx for training and validation
    train_data["id"] = np.arange(len(train_data))
    vali_data["id"] = np.arange(len(vali_data))
    train_idx = _get_valid_sequence(train_data, print_progress=config.verbose, previous_day=config.previous_day)
    vali_idx = _get_valid_sequence(vali_data, print_progress=config.verbose, previous_day=config.previous_day)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # get the empirical visit for sampling
    emp_visits = np.zeros(config["total_loc_num"])
    visits = train_data["location_id"].value_counts()
    for i, loc in enumerate(visits.index):
        emp_visits[loc] = visits.iloc[i]
    emp_visits = emp_visits / emp_visits.sum()

    # init models
    embedding = AllEmbedding(config=config).to(device)
    generator = Generator(
        device=device, config=config, embedding=embedding, starting_sample="real", starting_dist=emp_visits
    ).to(device)
    discriminator = Discriminator(config=config, embedding=embedding).to(device)
    # calculate parameters
    total_params_embed = sum(p.numel() for p in embedding.parameters() if p.requires_grad)
    total_params_generator = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total_params_discriminator = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(
        f"#Parameters embeddings: {total_params_embed} \t generator: {total_params_generator - total_params_embed} \t discriminator: {total_params_discriminator - total_params_embed}"
    )

    if not config.use_pretrain:
        log_dir = init_save_path(config, postfix="pretrain")

        discriminator, generator = pre_training(
            discriminator,
            generator,
            all_locs,
            config,
            device,
            log_dir,
            input_data=(train_data, train_idx, vali_data, vali_idx),
        )

    else:
        generator.load_state_dict(torch.load(os.path.join(config.save_root, config.pretrain_filepath, "generator.pt")))
        # discriminator.load_state_dict(
        #     torch.load(os.path.join(config.save_root, config.pretrain_filepath, "discriminator.pt"))
        # )

    print("Advtrain generator and discriminator ...")
    log_dir = init_save_path(config)
    adv_training(
        discriminator,
        generator,
        config,
        device,
        all_locs,
        log_dir,
        input_data=(train_data, train_idx, vali_data, vali_idx),
    )
