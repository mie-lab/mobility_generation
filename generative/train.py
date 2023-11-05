import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
from tqdm import tqdm

import os
from pathlib import Path
import pickle as pickle

import time

from trackintel.geogr.distances import calculate_distance_matrix

from generative.dataloader import (
    discriminator_dataset,
    discriminator_collate_fn,
    generator_dataset,
    generator_collate_fn,
    construct_discriminator_pretrain_dataset,
)
from utils.earlystopping import EarlyStopping


def generate_samples(model, config):
    samples = []
    for _ in range(int(config.num_gen_samples / config.d_batch_size)):
        samples.extend(model.sample(config.d_batch_size, config.generate_len).cpu().data.numpy().tolist())
    return np.array(samples)


def pre_training(discriminator, generator, all_locs, config, device, log_dir, input_data):
    train_data, train_idx, vali_data, vali_idx = input_data

    # pretrain discriminator
    fake_train_samples = construct_discriminator_pretrain_dataset(config, train_data, train_idx, all_locs)
    fake_vali_samples = construct_discriminator_pretrain_dataset(config, vali_data, vali_idx, all_locs)
    print("Pretrain discriminator")

    # train dataset
    d_train_data = discriminator_dataset(
        true_data=train_data, fake_data=fake_train_samples, valid_start_end_idx=train_idx
    )
    kwds_train = {
        "shuffle": True,
        "num_workers": config.num_workers,
        "batch_size": config.d_batch_size,
        "pin_memory": True,
    }
    d_train_loader = torch.utils.data.DataLoader(d_train_data, collate_fn=discriminator_collate_fn, **kwds_train)

    # validation dataset
    d_vali_data = discriminator_dataset(true_data=vali_data, fake_data=fake_vali_samples, valid_start_end_idx=vali_idx)
    kwds_vali = {
        "shuffle": False,
        "num_workers": config.num_workers,
        "batch_size": config.d_batch_size,
        "pin_memory": True,
    }
    d_vali_loader = torch.utils.data.DataLoader(d_vali_data, collate_fn=discriminator_collate_fn, **kwds_vali)

    d_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.pre_lr, weight_decay=config.weight_decay)

    print(f"length of the train loader: {len(d_train_loader)}\t #samples: {len(d_train_data)}")
    print(f"length of the validation loader: {len(d_vali_loader)}\t #samples: {len(d_vali_data)}")

    discriminator = training(
        d_train_loader,
        d_vali_loader,
        d_optimizer,
        d_criterion,
        discriminator,
        config,
        device,
        log_dir,
        model_type="discriminator",
    )

    # pretrain generator
    print("Pretrain generator")

    # training dataset
    g_train_data = generator_dataset(input_data=train_data, valid_start_end_idx=train_idx)
    kwds_train["batch_size"] = config.g_batch_size
    g_train_loader = torch.utils.data.DataLoader(g_train_data, collate_fn=generator_collate_fn, **kwds_train)

    g_vali_data = generator_dataset(input_data=vali_data, valid_start_end_idx=vali_idx)
    kwds_vali["batch_size"] = config.g_batch_size
    g_vali_loader = torch.utils.data.DataLoader(g_vali_data, collate_fn=generator_collate_fn, **kwds_vali)

    g_criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=config.pre_lr, weight_decay=config.weight_decay)

    print(f"length of the train loader: {len(g_train_loader)}\t #samples: {len(g_train_data)}")

    generator = training(
        g_train_loader,
        g_vali_loader,
        g_optimizer,
        g_criterion,
        generator,
        config,
        device,
        log_dir,
        model_type="generator",
    )

    return discriminator, generator


def training(
    train_loader,
    val_loader,
    optimizer,
    criterion,
    model,
    config,
    device,
    log_dir,
    model_type="discriminator",
):
    scheduler = StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    if config.verbose:
        print("Current learning rate: ", scheduler.get_last_lr()[0])

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        log_dir, patience=config.patience, verbose=config.verbose, delta=0.001, save_name=model_type
    )

    # Time for printing
    training_start_time = time.time()
    scheduler_count = 0

    # Loop for n_epochs
    for epoch in range(config.max_epoch):
        # train for one epoch
        train_epoch(config, model, train_loader, optimizer, criterion, device, epoch, model_type)

        # At the end of the epoch, do a pass on the validation set
        return_dict = validate_epoch(config, model, val_loader, criterion, device, model_type)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(return_dict, model)

        if early_stopping.early_stop:
            if config.verbose:
                print("=" * 50)
                print("Early stopping")
                print("Current learning rate: {:.6f}".format(optimizer.param_groups[0]["lr"]))
            if scheduler_count == 0:
                # early_stopping.best_return_dict
                print("Training finished.\t Time: {:.2f}s.".format((time.time() - training_start_time)))
                break

            scheduler_count += 1
            model.load_state_dict(torch.load(log_dir + f"/{model_type}.pt"))
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler.step()

        if config.debug is True:
            break

    # return model, performance

    return model


def train_epoch(config, model, data_loader, optimizer, criterion, device, epoch, model_type):
    n_batches = len(data_loader)

    running_loss = 0.0

    start_time = time.time()
    optimizer.zero_grad()
    for i, (data, target) in enumerate(data_loader):
        data = data.long().to(device).transpose(0, 1)
        target = target.to(device)

        pred = model(data)

        if model_type == "discriminator":
            loss = criterion(pred.view((-1,)), target.float().view((-1,)))
        else:
            loss = criterion(pred, target.long().view((-1,)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (config.verbose) and ((i + 1) % config["print_step"] == 0):
            print(
                "{}: Epoch {}, {:.1f}%\t loss: {:.3f}, took: {:.2f}s \r".format(
                    model_type,
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

    if config.verbose:
        print("")


def validate_epoch(config, model, data_loader, criterion, device, model_type):
    total_val_loss = 0

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data = data.long().to(device).transpose(0, 1)
            target = target.to(device)

            pred = model(data)

            if model_type == "discriminator":
                loss = criterion(pred.view((-1,)), target.float().view((-1,)))
            else:
                loss = criterion(pred, target.long().view((-1,)))

            total_val_loss += loss.item()

    # loss
    val_loss = total_val_loss / len(data_loader)

    if config.verbose:
        print("Validation loss = {:.2f}".format(val_loss))

    return {"val_loss": val_loss}
