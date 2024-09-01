# coding=utf-8
import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

from datetime import datetime
import pickle as pickle

from shapely import wkt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from easydict import EasyDict as edict

from utils.utils import load_data, setup_seed, load_config, init_save_path, get_train_test

from generative.movesim import Discriminator, Generator
from generative.train import pre_training, adv_training
from loc_predict.dataloader import load_data_predict
from utils.dist_util import load_state_dict
import blobfile as bf


def is_main_process():
    return get_rank() == 0


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def main(
    rank,
    world_size,
    config,
    all_locs,
    train_data,
    vali_data,
    emp_visits,
    # dist_matrix,
    # emp_matrix,
    # fct_matrix,
    log_dir,
):
    # setup the process groups
    setup(rank, world_size)

    # init models
    generator = Generator(
        device=rank,
        config=config,
        starting_sample="real",
        # dist_matrix=dist_matrix,
        # emp_matrix=emp_matrix,
        # fct_matrix=fct_matrix,
        starting_dist=emp_visits,
    ).to(rank)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    # find_unused_parameters=True for GAN training
    generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    generator = DDP(
        generator,
        device_ids=[rank],
        # find_unused_parameters=True,
    )

    # find_unused_parameters=True for GAN training
    discriminator = Discriminator(config=config).to(rank)
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
    discriminator = DDP(
        discriminator,
        device_ids=[rank],
        # find_unused_parameters=True,
    )
    # calculate parameters
    total_params_embed = sum(p.numel() for p in generator.module.embedding.parameters() if p.requires_grad)
    total_params_generator = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total_params_discriminator = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)

    if rank == 0:
        print(
            f"#Paras embeds: {total_params_embed} \t G: {total_params_generator - total_params_embed} \t D: {total_params_discriminator - total_params_embed}"
        )

    if not config.use_pretrain:
        discriminator, generator = pre_training(
            discriminator,
            generator,
            config,
            world_size,
            rank,
            log_dir,
            input_data=(train_data, vali_data),
        )

    else:
        checkpoint = load_state_dict(bf.join(config.save_root, config.pretrain_filepath, "generator.pt"))
        generator.load_state_dict(checkpoint["model"])

        checkpoint = load_state_dict(bf.join(config.save_root, config.pretrain_filepath, "discriminator.pt"))
        discriminator.load_state_dict(checkpoint["model"])

    if rank == 0:
        print("Advtrain generator and discriminator ...")

    adv_training(
        discriminator,
        generator,
        config,
        world_size,
        rank,
        all_locs,
        log_dir,
        input_data=(train_data, vali_data),
    )

    cleanup()


def get_data_for_emp(type):
    # read and preprocess
    sp = pd.read_csv(os.path.join(f"./data/sp_{type}.csv"), index_col="id")
    loc = pd.read_csv(os.path.join("./data/loc_s2_level10_14.csv"), index_col="id")

    sp = load_data(sp, loc)

    # get all possible locations
    all_locs = pd.read_csv("./data/s2_loc_visited_level10_14.csv", index_col="id")
    all_locs["geometry"] = all_locs["geometry"].apply(wkt.loads)
    all_locs = gpd.GeoDataFrame(all_locs, geometry="geometry", crs="EPSG:4326")
    # transform to projected coordinate systems
    all_locs = all_locs.to_crs("EPSG:2056")

    train_data, vali_data, test_data, all_locs = get_train_test(sp, all_locs=all_locs)

    print(
        f"Max loc id {all_locs.loc_id.max()}, min loc id {all_locs.loc_id.min()}, unique loc id:{all_locs.loc_id.unique().shape[0]}"
    )

    return train_data, vali_data, test_data, all_locs


if __name__ == "__main__":
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

    setup_seed(config.seed)

    world_size = config.world_size
    log_dir = init_save_path(config, time_now=int(datetime.now().timestamp()))

    # get the empirical visit for sampling
    train_df, _, _, all_locs = get_data_for_emp(type=config.dataset_variation)
    emp_visits = np.zeros(config["max_location"])
    visits = train_df["location_id"].value_counts()
    for i, loc in enumerate(visits.index):
        emp_visits[loc] = visits.iloc[i]
    emp_visits = emp_visits / emp_visits.sum()

    data_train = load_data_predict(batch_size=1, data_args=config, shuffle=True)
    data_valid = load_data_predict(batch_size=1, data_args=config, split="valid", shuffle=False)

    # distance and empirical visits
    # dist_matrix = pickle.load(open(os.path.join(config.temp_save_root, "matrix", "distance_13.pk"), "rb")).astype(
    #     np.float32
    # )
    # emp_matrix = pickle.load(open(os.path.join(config.temp_save_root, "matrix", "visits_13.pk"), "rb")).astype(
    #     np.float32
    # )
    # fct_matrix = pickle.load(open(os.path.join(config.temp_save_root, "matrix", "function_13.pk"), "rb")).astype(
    #     np.float32
    # )

    mp.spawn(
        main,
        args=(
            world_size,
            config,
            all_locs,
            data_train,
            data_valid,
            emp_visits,
            # dist_matrix,
            # emp_matrix,
            # fct_matrix,
            log_dir,
        ),
        nprocs=world_size,
    )
