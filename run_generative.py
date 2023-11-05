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

from sklearn.preprocessing import OrdinalEncoder

from trackintel.geogr.distances import calculate_distance_matrix

from utils.utils import load_data, setup_seed, load_config, init_save_path
from utils.dataloader import collate_fn, get_train_test, _get_valid_sequence
from loc_predict.models.markov import markov_transition_prob

from generative.movesim import Discriminator, Generator, AllEmbedding
from generative.rollout import Rollout
from generative.gan_loss import GANLoss, periodLoss, distanceLoss
from generative.train import generate_samples, pre_training, train_epoch, validate_epoch
from generative.dataloader import (
    discriminator_dataset,
    discriminator_collate_fn,
    generator_dataset,
    generator_collate_fn,
    construct_discriminator_pretrain_dataset,
)


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

    train_data, vali_data, test_data, all_locs = get_train_test(sp, all_locs=all_locs)
    print(f"Max location id:{all_locs.loc_id.max()}, unique location id:{all_locs.loc_id.unique().shape[0]}")
    config["total_loc_num"] = int(all_locs.loc_id.max() + 1)
    config["total_user_num"] = int(train_data.user_id.max() + 1)

    # get valid idx for training and validation
    train_data["id"] = np.arange(len(train_data))
    vali_data["id"] = np.arange(len(vali_data))
    train_idx = _get_valid_sequence(train_data, print_progress=config.verbose)
    vali_idx = _get_valid_sequence(vali_data, print_progress=config.verbose)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # embeddings
    embedding = AllEmbedding(config=config).to(device)
    generator = Generator(device=device, config=config, embedding=embedding).to(device)
    discriminator = Discriminator(config=config, embedding=embedding).to(device)
    #
    total_params_embed = sum(p.numel() for p in embedding.parameters() if p.requires_grad)
    total_params_generator = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total_params_discriminator = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(
        f"#Parameters embeddings: {total_params_embed} \t generator: {total_params_generator - total_params_embed} \t discriminator: {total_params_discriminator - total_params_embed}"
    )

    # transit_df = train_data.groupby("user_id").apply(markov_transition_prob, n=1).reset_index()
    # transit_df = transit_df.groupby(["loc_1", "toLoc"])["size"].sum().reset_index()
    # # transition_df[["", "", ""]]

    # print(transit_df)

    # pdistance = calculate_distance_matrix(all_locs, all_locs.iloc[0:1], dist_metric="haversine")
    # print(pdistance)
    # opt = parser.parse_args()
    # main(opt)

    if not config.use_pretrain:
        log_dir = init_save_path(config)

        discriminator, generator = pre_training(
            discriminator,
            generator,
            all_locs,
            config,
            device,
            log_dir,
            input_data=(train_data, train_idx, vali_data, vali_idx),
        )

    # else:
    #     generator.load_state_dict(torch.load(os.path.join(config.save_root, config.pretrain_filepath, "generator.pt")))
    #     discriminator.load_state_dict(
    #         torch.load(os.path.join(config.save_root, config.pretrain_filepath, "discriminator.pt"))
    #     )

    print("Advtrain generator and discriminator ...")
    rollout = Rollout(generator, 0.9)

    # gan loss and optimizer
    gen_gan_loss = GANLoss().to(device)
    gen_gan_optm = optim.Adam(generator.parameters(), lr=config.g_lr, weight_decay=config.weight_decay)
    # period loss
    period_crit = periodLoss(time_interval=24).to(device)
    # distance loss
    distance_crit = distanceLoss(locations=all_locs, device=device).to(device)

    d_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.d_lr, weight_decay=config.weight_decay)

    running_loss = 0.0
    for epoch in range(config.max_epoch):
        # Train the generator for one step
        print("training generator")
        start_time = time.time()

        iterations = range(3) if epoch != 0 else range(1)
        for it in iterations:
            samples = generator.sample(config.d_batch_size, config.generate_len)
            # construct the input to the generator, add zeros before samples and delete the last column
            zeros = torch.zeros((config.d_batch_size, 1)).long().to(device)

            inputs = torch.cat([zeros, samples], dim=1)[:, :-1]

            targets = samples.view((-1,))

            # calculate the reward
            rewards = rollout.get_reward(samples, roll_out_num=8, discriminator=discriminator, device=device)

            prob = generator.forward(inputs)

            gloss = gen_gan_loss(prob, targets, rewards, device)

            # additional losses
            gloss += config.periodic_loss * period_crit(samples)
            gloss += config.distance_loss * distance_crit(samples)
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
                true_data=train_data, fake_data=generated_samples, valid_start_end_idx=train_idx
            )
            kwds_train = {
                "shuffle": True,
                "num_workers": config.num_workers,
                "batch_size": config.d_batch_size,
                "pin_memory": True,
            }
            d_train_loader = torch.utils.data.DataLoader(
                d_train_data, collate_fn=discriminator_collate_fn, **kwds_train
            )

            for i in range(2):
                train_epoch(
                    config,
                    discriminator,
                    d_train_loader,
                    d_optimizer,
                    d_criterion,
                    device,
                    epoch=i,
                    model_type="discriminator",
                )
