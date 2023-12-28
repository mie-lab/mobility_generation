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

from utils.utils import load_data, setup_seed, load_config, init_save_path
from utils.dataloader import get_train_test, _get_valid_sequence

from generative.movesim import Discriminator, Generator
from generative.train import pre_training, adv_training


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
    train_idx,
    vali_idx,
    emp_visits,
    dist_matrix,
    emp_matrix,
    fct_matrix,
    log_dir,
):
    # setup the process groups
    setup(rank, world_size)

    # init models
    generator = Generator(
        device=rank,
        config=config,
        starting_sample="real",
        dist_matrix=dist_matrix,
        emp_matrix=emp_matrix,
        fct_matrix=fct_matrix,
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
    total_params_embed = sum(p.numel() for p in generator.module.loc_embedding.parameters() if p.requires_grad)
    total_params_generator = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total_params_discriminator = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)

    if rank == 0:
        print(
            f"#Parameters embeddings: {total_params_embed} \t generator: {total_params_generator - total_params_embed} \t discriminator: {total_params_discriminator - total_params_embed}"
        )

    if not config.use_pretrain:
        discriminator, generator = pre_training(
            discriminator,
            generator,
            all_locs,
            config,
            world_size,
            rank,
            log_dir,
            input_data=(train_data, train_idx, vali_data, vali_idx),
        )

    else:
        generator.load_state_dict(
            torch.load(os.path.join(config.save_root, config.pretrain_filepath, "generator_pretrain.pt"))
        )
        discriminator.load_state_dict(
            torch.load(os.path.join(config.save_root, config.pretrain_filepath, "discriminator_pretrain.pt"))
        )

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
        input_data=(train_data, train_idx, vali_data, vali_idx),
    )

    cleanup()


def preprocess_datasets(config, dataset="sp"):
    # read and preprocess
    sp = pd.read_csv(os.path.join(config.temp_save_root, f"{dataset}.csv"), index_col="id")
    loc = pd.read_csv(os.path.join(config.temp_save_root, "locs_s2.csv"), index_col="id")
    sp = load_data(sp, loc)

    all_locs = pd.read_csv(os.path.join(config.temp_save_root, "test", "all_locations.csv"), index_col="id")
    all_locs["geometry"] = all_locs["geometry"].apply(wkt.loads)
    all_locs = gpd.GeoDataFrame(all_locs, geometry="geometry", crs="EPSG:4326")
    # transform to projected coordinate systems
    all_locs = all_locs.to_crs("EPSG:2056")

    train_data, vali_data, test_data, all_locs = get_train_test(sp, all_locs=all_locs)

    train_data["id"] = np.arange(len(train_data))
    vali_data["id"] = np.arange(len(vali_data))
    test_data["id"] = np.arange(len(test_data))

    return train_data, vali_data, test_data, all_locs


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

    world_size = config.world_size
    log_dir = init_save_path(config, time_now=int(datetime.now().timestamp()))

    train_data, vali_data, test_data, all_locs = preprocess_datasets(config)

    print(f"Max location id:{all_locs.loc_id.max()}, unique location id:{all_locs.loc_id.unique().shape[0]}")
    config["total_loc_num"] = int(all_locs.loc_id.max() + 1)
    config["total_user_num"] = int(train_data.user_id.max() + 1)

    # get valid idx for training and validation
    train_idx = _get_valid_sequence(train_data, print_progress=config.verbose, previous_day=config.previous_day)
    vali_idx = _get_valid_sequence(vali_data, print_progress=config.verbose, previous_day=config.previous_day)
    # test_idx = _get_valid_sequence(test_data, print_progress=config.verbose, previous_day=config.previous_day)

    # get the empirical visit for sampling
    emp_visits = np.zeros(config["total_loc_num"])
    visits = train_data["location_id"].value_counts()
    for i, loc in enumerate(visits.index):
        emp_visits[loc] = visits.iloc[i]
    emp_visits = emp_visits / emp_visits.sum()

    # distance and empirical visits
    dist_matrix = pickle.load(open(os.path.join(config.temp_save_root, "temp", "dist_matrix_test.pk"), "rb"))
    emp_matrix = pickle.load(open(os.path.join(config.temp_save_root, "temp", "emp_matrix_test.pk"), "rb"))
    fct_matrix = pickle.load(open(os.path.join(config.temp_save_root, "temp", "function_matrix_test.pk"), "rb"))

    mp.spawn(
        main,
        args=(
            world_size,
            config,
            all_locs,
            train_data,
            vali_data,
            train_idx,
            vali_idx,
            emp_visits,
            dist_matrix,
            emp_matrix,
            fct_matrix,
            log_dir,
        ),
        nprocs=world_size,
    )
