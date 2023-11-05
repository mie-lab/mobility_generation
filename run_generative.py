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
from generative.train import generate_samples, construct_discriminator_pretrain_dataset
from generative.dataloader import discriminator_dataset, discriminator_collate_fn


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
    sp = pd.read_csv(os.path.join(config.temp_save_root, "sp_small.csv"), index_col="id")
    loc = pd.read_csv(os.path.join(config.temp_save_root, "locs_s2.csv"), index_col="id")
    sp = load_data(sp, loc)

    all_locs = pd.read_csv(os.path.join(config.temp_save_root, "all_locations.csv"), index_col="id")
    all_locs["geometry"] = all_locs["geometry"].apply(wkt.loads)
    all_locs = gpd.GeoDataFrame(all_locs, geometry="geometry", crs="EPSG:4326")

    train_data, vali_data, test_data, all_locs = get_train_test(sp, all_locs=all_locs)
    print(f"Max location id:{all_locs.loc_id.max()}, unique location id:{all_locs.loc_id.unique().shape[0]}")
    config["total_loc_num"] = int(all_locs.loc_id.max() + 1)
    config["total_user_num"] = int(train_data.user_id.max() + 1)

    # get valid idx for training
    train_data["id"] = np.arange(len(train_data))
    train_idx = _get_valid_sequence(train_data, print_progress=config.verbose)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # location embeddings
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

    if not config.use_pretrain:
        log_dir = init_save_path(config)

        # pretrain discriminator
        fake_samples = construct_discriminator_pretrain_dataset(config, train_data, train_idx, all_locs)
        print("Pretrain discriminator")
        d_train_data = discriminator_dataset(
            true_data=train_data, fake_data=fake_samples, valid_start_end_idx=train_idx
        )
        kwds_train = {
            "shuffle": True,
            "num_workers": config["num_workers"],
            "batch_size": config["batch_size"],
            "pin_memory": True,
        }
        d_train_loader = torch.utils.data.DataLoader(d_train_data, collate_fn=discriminator_collate_fn, **kwds_train)

        dis_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(device)
        dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.000001)

        running_loss = 0.0
        n_batches = len(d_train_loader)
        print(f"length of the train loader: {n_batches}\t #samples: {len(d_train_data)}")
        for epoch in range(10):
            total_loss = 0.0
            start_time = time.time()
            dis_optimizer.zero_grad()
            for i, (data, target) in enumerate(d_train_loader):
                data = data.long().to(device).transpose(0, 1)
                target = target.float().to(device)

                pred = discriminator(data)

                loss = dis_criterion(pred.view((-1,)), target.view((-1,)))

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

            dloss = total_loss / n_batches
            if config.verbose:
                print("")
                print(f"Epoch [{epoch}], Discriminator Loss: {dloss:.2f}")

        # pretrain generator
        print("Pretrain generator")
        gen_data_iter = NewGenIter(REAL_DATA, BATCH_SIZE)

        gen_criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
        gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)

        pretrain_model(
            "G", g_pre_epoch, generator, gen_data_iter, gen_criterion, gen_optimizer, BATCH_SIZE, device=device
        )

        # save networks
        torch.save(generator.state_dict(), os.path.join(log_dir, "generator.pt"))
        torch.save(discriminator.state_dict(), os.path.join(log_dir, "discriminator.pt"))

    else:
        generator.load_state_dict(torch.load(os.path.join(config.save_root, config.pretrain_filepath, "generator.pt")))
        discriminator.load_state_dict(
            torch.load(os.path.join(config.save_root, config.pretrain_filepath, "discriminator.pt"))
        )

    print("advtrain generator and discriminator ...")
    rollout = Rollout(generator, 0.8)

    # gan loss and optimizer
    gen_gan_loss = GANLoss().to(device)
    gen_gan_optm = optim.Adam(generator.parameters(), lr=0.0001)
    # period loss
    period_crit = periodLoss(time_interval=24).to(device)
    # distance loss
    distance_crit = distanceLoss(locations=all_locs, device=device).to(device)

    dis_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(device)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)

    running_loss = 0.0
    for epoch in range(config.max_epoch):
        # Train the generator for one step
        print("training generator")
        start_time = time.time()

        iterations = range(3) if epoch != 0 else range(1)
        for it in iterations:
            samples = generator.sample(config.batch_size, config.generate_len)
            # construct the input to the generator, add zeros before samples and delete the last column
            zeros = torch.zeros((config.batch_size, 1)).long().to(device)

            inputs = torch.cat([zeros, samples], dim=1)[:, :-1]

            # time_tensor = torch.LongTensor([i % 24 for i in range(config.generate_len)]).to(device)
            # time_tensor = time_tensor.repeat(config.batch_size).reshape(config.batch_size, -1)

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
                "num_workers": config["num_workers"],
                "batch_size": config["batch_size"],
                "pin_memory": True,
            }
            d_train_loader = torch.utils.data.DataLoader(
                d_train_data, collate_fn=discriminator_collate_fn, **kwds_train
            )

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

                dloss = total_loss / len(d_train_loader)
            if config.verbose:
                print("")

        print(f"Epoch [{epoch}] Generator Loss: {gloss.item():.2f}, Discriminator Loss: {dloss:.2f}")
