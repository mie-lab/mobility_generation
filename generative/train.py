import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import os
from pathlib import Path
import pickle as pickle

import time

from generative.gan_dataloader import (
    discriminator_dataset,
    discriminator_collate_fn,
    generator_dataset,
    generator_collate_fn,
    construct_discriminator_pretrain_dataset,
)
from generative.rollout import Rollout
from generative.gan_loss import GANLoss, periodLoss, distanceLoss
from metrics.evaluations import Metric
from utils.earlystopping import EarlyStopping


def generate_samples(model, config, num, single_len=256, print_progress=False):
    samples = []
    for _ in tqdm(range(int(num / single_len)), disable=not print_progress):
        samples.extend(model.sample(single_len, config.generate_len).cpu().data.numpy().tolist())
    return np.array(samples)


def pre_training(discriminator, generator, all_locs, config, device, log_dir, input_data):
    train_data, train_idx, vali_data, vali_idx = input_data

    # pretrain discriminator
    fake_train_samples = construct_discriminator_pretrain_dataset(train_data, train_idx, all_locs)
    fake_vali_samples = construct_discriminator_pretrain_dataset(vali_data, vali_idx, all_locs)
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

    d_criterion = nn.NLLLoss(reduction="mean").to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.pre_lr, weight_decay=config.weight_decay)

    print(f"length of the train loader: {len(d_train_loader)}\t #samples: {len(d_train_data)}")
    print(f"length of the validation loader: {len(d_vali_loader)}\t #samples: {len(d_vali_data)}")

    # discriminator = training(
    #     d_train_loader,
    #     d_vali_loader,
    #     d_optimizer,
    #     d_criterion,
    #     discriminator,
    #     config,
    #     device,
    #     log_dir,
    #     model_type="discriminator",
    # )

    # pretrain generator
    print("Pretrain generator")

    # training dataset
    g_train_data = generator_dataset(input_data=train_data, valid_start_end_idx=train_idx)
    kwds_train["batch_size"] = config.g_batch_size
    g_train_loader = torch.utils.data.DataLoader(g_train_data, collate_fn=generator_collate_fn, **kwds_train)

    g_vali_data = generator_dataset(input_data=vali_data, valid_start_end_idx=vali_idx)
    kwds_vali["batch_size"] = config.g_batch_size
    g_vali_loader = torch.utils.data.DataLoader(g_vali_data, collate_fn=generator_collate_fn, **kwds_vali)

    g_criterion = nn.NLLLoss(reduction="mean").to(device)
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
            if scheduler_count == 2:
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
    training_loss = 0

    start_time = time.time()
    optimizer.zero_grad()
    for i, (data, target) in enumerate(data_loader):
        data = data.long().to(device).transpose(0, 1)
        target = target.to(device)

        pred = model(data)

        if model_type == "discriminator":
            loss = criterion(pred, target.long().view((-1,)))
        else:
            loss = criterion(pred, target.long().view((-1,)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        training_loss += loss.item()
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
        print("Training loss = {:.5f}".format(training_loss / n_batches))


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
        print("Validation loss = {:.5f}".format(val_loss))

    return {"val_loss": val_loss}


def adv_training(discriminator, generator, config, device, all_locs, log_dir, input_data):
    train_data, train_idx, vali_data, vali_idx = input_data

    rollout = Rollout(generator, 0.9)

    # gan loss and optimizer
    gen_gan_loss = GANLoss().to(device)
    gen_gan_optm = optim.Adam(generator.parameters(), lr=config.g_lr, weight_decay=config.weight_decay)
    # period loss
    # period_crit = periodLoss(time_interval=24).to(device)
    # distance loss
    # distance_crit = distanceLoss(locations=all_locs, device=device).to(device)

    # evaluation
    metrics = Metric(config, locations=all_locs, input_data=vali_data, valid_start_end_idx=vali_idx)

    d_criterion = nn.NLLLoss(reduction="mean").to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.d_lr, weight_decay=config.weight_decay)

    for epoch in range(config.adv_max_epoch):
        # Train the generator for one step
        print("training generator")
        start_time = time.time()

        # train
        start_time = time.time()

        # evaluate current generator performance
        samples = generate_samples(generator, config, single_len=config.d_batch_size, num=config.d_batch_size)
        jsds = metrics.get_individual_jsds(gene_data=samples)
        print(
            "Metric: distance {:.3f}, rg {:.3f}, period {:.3f}, topk all {:.3f}, topk {:.3f}".format(
                jsds[0], jsds[1], jsds[2], jsds[3], jsds[4]
            )
        )

        # only once and update rollout parameter
        train_loss = train_generator(
            generator,
            discriminator,
            samples,
            rollout,
            gen_gan_loss,
            gen_gan_optm,
            config,
            device,
            # crit=(period_crit, distance_crit),
        )
        rollout.update_params()

        if config.verbose:
            print(
                "Generator: Epoch {}\t loss: {:.3f}, took: {:.2f}s \r".format(
                    epoch + 1,
                    train_loss,
                    time.time() - start_time,
                )
            )

        # train discriminator
        print("training discriminator")
        for _ in range(1):
            samples = generate_samples(
                generator, config, num=config.num_gen_samples, single_len=2048, print_progress=False
            )
            d_train_data = discriminator_dataset(true_data=train_data, fake_data=samples, valid_start_end_idx=train_idx)
            kwds_train = {
                "shuffle": True,
                "num_workers": config.num_workers,
                "batch_size": config.d_batch_size,
                "pin_memory": True,
            }
            d_train_loader = torch.utils.data.DataLoader(
                d_train_data, collate_fn=discriminator_collate_fn, **kwds_train
            )

            for i in range(1):
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

        torch.save(generator.state_dict(), log_dir + "/generator.pt")
        torch.save(discriminator.state_dict(), log_dir + "/discriminator.pt")

        if epoch > 10:
            samples = generate_samples(
                generator, config, num=config.num_gen_samples, single_len=1024, print_progress=False
            )
            save_path = os.path.join(config.temp_save_root, "temp", f"generated_samples_{epoch}.pk")
            save_pk_file(save_path, samples)


def train_generator(generator, discriminator, samples, rollout, gen_gan_loss, gen_gan_optm, config, device, crit=None):
    # period_crit, distance_crit = crit

    # construct the input to the generator, add zeros before samples and delete the last column
    zeros = torch.zeros((config.d_batch_size, 1)).long().to(device)
    samples = torch.Tensor(samples).long().to(device)

    inputs = torch.cat([zeros, samples], dim=1)[:, :-1]
    targets = samples.view((-1,))

    # calculate the reward
    rewards = rollout.get_reward(samples, roll_out_num=config.rollout_num, discriminator=discriminator, device=device)
    rewards = torch.exp(rewards)

    prob = generator.forward(inputs)

    gloss = gen_gan_loss(prob, targets, rewards, device)

    # additional losses
    # gloss += config.periodic_loss * period_crit(samples)
    # gloss += config.distance_loss * distance_crit(samples)

    running_loss = gloss.item()
    # optimize
    gen_gan_optm.zero_grad()
    gloss.backward()
    gen_gan_optm.step()

    return running_loss


def save_pk_file(save_path, data):
    """Function to save data to pickle format given data and path."""
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
