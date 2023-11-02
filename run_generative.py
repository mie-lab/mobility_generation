# coding=utf-8
import pdb
import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm

from torch import nn, optim
import time

from shapely import wkt, Point


from easydict import EasyDict as edict

from scipy.spatial.distance import pdist, squareform

from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OrdinalEncoder

from trackintel.geogr.distances import calculate_distance_matrix

from utils.utils import load_data, setup_seed, load_config
from utils.dataloader import collate_fn, _split_dataset, traj_dataset
from loc_predict.models.markov import markov_transition_prob
from generative.movesim import Discriminator, Generator, AllEmbedding
from generative.rollout import Rollout
from generative.gan_loss import GANLoss
from generative.train import generate_samples
from generative.dataloader import discriminator_dataset


def _get_train_test(sp, all_locs):
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp.drop(columns={"started_at", "finished_at"}, inplace=True)

    # encoder user, 0 reserved for padding
    enc = OrdinalEncoder(dtype=np.int64)
    sp["user_id"] = enc.fit_transform(sp["user_id"].values.reshape(-1, 1)) + 1

    # truncate too long duration, >2 days to 2 days
    sp["duration"] = sp["duration"] / 60
    sp.loc[sp["duration"] > 60 * 24 * 2 - 1, "duration"] = 60 * 24 * 2 - 1

    # split the datasets, user dependent 0.6, 0.2, 0.2
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


def get_dataloaders(train_data, vali_data, test_data, config):
    dataset_train = traj_dataset(train_data, print_progress=config.verbose)
    dataset_val = traj_dataset(vali_data, print_progress=config.verbose)
    dataset_test = traj_dataset(test_data, print_progress=config.verbose)

    kwds_train = {
        "shuffle": True,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": True,
    }
    kwds_val = {
        "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": True,
    }
    kwds_test = {
        "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": True,
    }

    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, **kwds_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=collate_fn, **kwds_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=collate_fn, **kwds_test)

    print(
        f"length of the train loader: {len(train_loader)}\t validation loader:{len(val_loader)}\t test loader:{len(test_loader)}"
    )
    return (
        train_loader,
        val_loader,
        test_loader,
    )


def dis_collate_fn(batch):
    """function to collate data samples into batch tensors."""
    src_batch, tgt_batch = [], []

    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)

    return src_batch, tgt_batch


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

    # print(sp)
    # print(all_locs)
    train_data, vali_data, test_data, all_locs = _get_train_test(sp, all_locs)
    print(f"Max location id:{all_locs.loc_id.max()}, unique location id:{all_locs.loc_id.unique().shape[0]}")
    config["total_loc_num"] = int(all_locs.loc_id.max() + 1)
    config["total_user_num"] = int(train_data.user_id.max() + 1)

    train_loader, val_loader, test_loader = get_dataloaders(train_data, vali_data, test_data, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding = AllEmbedding(config=config).to(device)

    generator = Generator(device=device, config=config, embedding=embedding).to(device)
    total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print("#Parameters generator: ", total_params)

    discriminator = Discriminator(config=config, embedding=embedding).to(device)
    total_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print("#Parameters discriminator: ", total_params)

    # transit_df = train_data.groupby("user_id").apply(markov_transition_prob, n=1).reset_index()
    # transit_df = transit_df.groupby(["loc_1", "toLoc"])["size"].sum().reset_index()
    # # transition_df[["", "", ""]]

    # print(transit_df)

    # pdistance = calculate_distance_matrix(all_locs, all_locs.iloc[0:1], dist_metric="haversine")
    # print(pdistance)
    # opt = parser.parse_args()
    # main(opt)

    rollout = Rollout(generator, 0.8)

    gen_gan_loss = GANLoss().to(device)
    gen_gan_optm = optim.Adam(generator.parameters(), lr=0.0001)

    dis_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(device)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)

    running_loss = 0.0
    for epoch in range(config.max_epoch):
        # Train the generator for one step
        print("training generator")
        start_time = time.time()

        iterations = range(5) if epoch != 0 else range(1)
        for it in iterations:
            samples = generator.sample(config.batch_size, config.generate_len)
            # construct the input to the generator, add zeros before samples and delete the last column
            zeros = torch.zeros((config.batch_size, 1)).long().to(device)

            inputs = torch.cat([zeros, samples], dim=1)[:, :-1]

            time_tensor = torch.LongTensor([i % 24 for i in range(config.generate_len)]).to(device)
            time_tensor = time_tensor.repeat(config.batch_size).reshape(config.batch_size, -1)

            targets = samples.view((-1,))

            # calculate the reward
            rewards = rollout.get_reward(samples, roll_out_num=2, discriminator=discriminator, device=device)

            prob = generator.forward(inputs, time_tensor)

            gloss = gen_gan_loss(prob, targets, rewards, device)

            # optimize
            gen_gan_optm.zero_grad()
            gloss.backward()
            gen_gan_optm.step()

            running_loss += gloss.item()
            if config.verbose:
                print(
                    "Generator: Epoch {}, train iter {}\t loss: {:.3f}, took: {:.2f}s \r".format(
                        epoch + 1,
                        it,
                        running_loss,
                        time.time() - start_time,
                    )
                )
                running_loss = 0.0
                start_time = time.time()

            rollout.update_params()

        print("training discriminator")

        for _ in range(1):
            generated_samples = generate_samples(generator, config)

            d_train_data = discriminator_dataset(
                true_data=train_data, fake_data=generated_samples, print_progress=False
            )
            kwds_train = {
                "shuffle": True,
                "num_workers": config["num_workers"],
                "batch_size": config["batch_size"],
                "pin_memory": True,
            }
            d_train_loader = torch.utils.data.DataLoader(d_train_data, collate_fn=dis_collate_fn, **kwds_train)

            n_batches = len(d_train_loader)
            for _ in range(2):
                total_loss = 0.0

                start_time = time.time()
                dis_optimizer.zero_grad()
                for i, (data, target) in enumerate(d_train_loader):
                    data = data.long().to(device).transpose(0, 1)
                    target = target.float().to(device)

                    pred = discriminator(data)

                    loss = dis_criterion(pred.view((-1,)), target.view((-1,)))
                    # loss = dis_criterion(pred, target)
                    total_loss += loss.item()
                    dis_optimizer.zero_grad()
                    loss.backward()
                    dis_optimizer.step()

                    running_loss += loss.item()
                    if (config.verbose) and ((i + 1) % config["print_step"] == 0):
                        print(
                            "Discriminator: Epoch {}, {:.1f}%\t loss: {:.3f}, took: {:.2f}s \r".format(
                                epoch + 1,
                                100 * (i + 1) / n_batches,
                                running_loss / config["print_step"],
                                time.time() - start_time,
                            ),
                            end="",
                            flush=True,
                        )
                        running_loss = 0.0
                        start_time = time.time()

                dloss = total_loss / (i + 1)

        print(f"Epoch [{epoch}] Generator Loss: {gloss.item():.2f}, Discriminator Loss: {dloss:.2f}")
