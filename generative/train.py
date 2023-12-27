import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist

import random
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


def generate_samples(model, seq_len, num, single_len=256, print_progress=False):
    samples = []
    for _ in tqdm(range(int(num / single_len)), disable=not print_progress):
        samples.extend(model.module.sample(single_len, seq_len).detach().cpu().numpy().tolist())
    return np.array(samples)


def get_pretrain_loaders(config, input_data, all_locs, world_size, device):
    train_data, train_idx, vali_data, vali_idx = input_data

    # train dataset
    fake_train_samples = construct_discriminator_pretrain_dataset(train_data, train_idx, all_locs)
    fake_vali_samples = construct_discriminator_pretrain_dataset(vali_data, vali_idx, all_locs)

    d_train_data = discriminator_dataset(
        true_data=train_data, fake_data=fake_train_samples, valid_start_end_idx=train_idx
    )
    train_sampler = DistributedSampler(d_train_data, num_replicas=world_size, rank=device, shuffle=True, drop_last=True)
    d_train_loader = torch.utils.data.DataLoader(
        d_train_data,
        collate_fn=discriminator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.d_batch_size,
        pin_memory=True,
        sampler=train_sampler,
    )

    # validation dataset
    d_vali_data = discriminator_dataset(true_data=vali_data, fake_data=fake_vali_samples, valid_start_end_idx=vali_idx)
    vali_sampler = DistributedSampler(d_vali_data, num_replicas=world_size, rank=device, shuffle=False)
    d_vali_loader = torch.utils.data.DataLoader(
        d_vali_data,
        collate_fn=discriminator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.d_batch_size,
        pin_memory=True,
        sampler=vali_sampler,
    )

    # training dataset
    g_train_data = generator_dataset(input_data=train_data, valid_start_end_idx=train_idx)
    train_sampler = DistributedSampler(g_train_data, num_replicas=world_size, rank=device, shuffle=True)
    g_train_loader = torch.utils.data.DataLoader(
        g_train_data,
        collate_fn=generator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.g_batch_size,
        pin_memory=True,
        sampler=train_sampler,
    )

    # validation dataset
    g_vali_data = generator_dataset(input_data=vali_data, valid_start_end_idx=vali_idx)
    vali_sampler = DistributedSampler(g_vali_data, num_replicas=world_size, rank=device, shuffle=False)
    g_vali_loader = torch.utils.data.DataLoader(
        g_vali_data,
        collate_fn=generator_collate_fn,
        num_workers=config.num_workers,
        batch_size=config.g_batch_size,
        pin_memory=True,
        sampler=vali_sampler,
    )

    if is_main_process():
        print(f"len d_train loader:\t{len(d_train_loader)}\t #samples: {len(d_train_data)}")
        print(f"len d_vali loader:\t{len(d_vali_loader)}\t #samples: {len(d_vali_data)}")
        print(f"len g_train loader:\t{len(g_train_loader)}\t #samples: {len(g_train_data)}")
        print(f"len g_vali loader:\t{len(g_vali_loader)}\t #samples: {len(g_vali_data)}")

    return d_train_loader, d_vali_loader, g_train_loader, g_vali_loader


def pre_training(discriminator, generator, all_locs, config, world_size, device, log_dir, input_data):
    d_train_loader, d_vali_loader, g_train_loader, g_vali_loader = get_pretrain_loaders(
        config, input_data, all_locs, world_size, device
    )

    # loss and optimizer
    d_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.pre_d_lr, weight_decay=config.weight_decay)
    g_criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=0).to(device)  # ignore_index for padding
    g_optimizer = optim.Adam(generator.parameters(), lr=config.pre_g_lr, weight_decay=config.weight_decay)

    # pretrain generator
    if is_main_process():
        print("Pretrain generator")

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

    # pretrain discriminator
    if is_main_process():
        print("Pretrain discriminator")
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
    scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_gamma)
    if config.verbose and is_main_process():
        print("Current learning rate: ", scheduler.get_last_lr()[0])

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        log_dir,
        patience=config.patience,
        verbose=config.verbose,
        main_process=is_main_process(),
        delta=0.001,
        save_name=model_type + "_pretrain",
    )

    # Time for printing
    training_start_time = time.time()
    scheduler_count = 0

    # Loop for n_epochs
    for epoch in range(config.max_epoch):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(config, model, train_loader, optimizer, criterion, device, epoch, model_type)

        # At the end of the epoch, do a pass on the validation set
        return_dict = validate_epoch(config, model, val_loader, criterion, device, model_type)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(return_dict, model)

        if early_stopping.early_stop:
            if config.verbose and is_main_process():
                print("=" * 50)
                print("Early stopping")
                print("Current learning rate: {:.6f}".format(optimizer.param_groups[0]["lr"]))
            if scheduler_count == 2:
                # early_stopping.best_return_dict
                if is_main_process():
                    print("Training finished.\t Time: {:.2f}s.".format((time.time() - training_start_time)))
                break

            # for multigpu
            dist.barrier()
            map_location = {"cuda:%d" % 0: "cuda:%d" % device}
            model.load_state_dict(torch.load(log_dir + f"/{model_type}_pretrain.pt", map_location=map_location))
            #
            scheduler_count += 1
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler.step()

        if config.debug is True:
            break

    return model


def train_epoch(config, model, data_loader, optimizer, criterion, device, epoch, model_type):
    n_batches = len(data_loader)

    model.train()

    running_loss = 0.0
    training_loss = 0
    all_correct = 0
    all_total = 0
    curr_correct = 0
    curr_total = 0

    start_time = time.time()
    optimizer.zero_grad()
    for i, (data, target) in enumerate(data_loader):
        data = data.long().to(device)
        target = target.to(device)

        pred = model(data)

        if model_type == "discriminator":
            loss = criterion(pred.view((-1,)), target.float().view((-1,)))

            # get the accuracy
            prob = torch.sigmoid(pred).view(-1)

            correct = torch.sum(torch.eq(prob > 0.5, target.to(bool)))
            total = len(target)

            all_correct += correct
            all_total += total
            curr_correct += correct
            curr_total += total

        else:
            loss = criterion(pred, target.long().view((-1,)))

            # get the accuracy
            prediction = torch.topk(pred, k=1, dim=-1).indices.view(-1)
            valid_idx = target.view(-1) != 0

            correct = torch.sum(torch.eq(prediction[valid_idx], target.view(-1)[valid_idx]))
            total = len(target.view(-1)[valid_idx])
            all_correct += correct
            all_total += total
            curr_correct += correct
            curr_total += total

        if torch.isnan(loss):
            assert False
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        running_loss += loss.item()
        training_loss += loss.item()
        if (config.verbose) and ((i + 1) % config["print_step"] == 0) and is_main_process():
            print(
                "{}: Epoch {}, {:.1f}%\t loss: {:.3f}, acc: {:.2f} took: {:.2f}s \r".format(
                    model_type,
                    epoch + 1,
                    100 * (i + 1) / n_batches,
                    running_loss / config["print_step"],
                    curr_correct * 100 / curr_total,
                    time.time() - start_time,
                ),
                end="",
                flush=True,
            )
            running_loss = 0.0
            curr_correct = 0
            curr_total = 0
            start_time = time.time()

    if config.verbose and is_main_process():
        print("")
        print(
            "Training loss = {:.5f}\t Accuracy = {:.2f}".format(
                training_loss / n_batches, all_correct * 100 / all_total
            )
        )


def validate_epoch(config, model, data_loader, criterion, device, model_type):
    total_val_loss = 0
    correct = 0
    total = 0

    # change to validation mode. Warning in inference mode regarding the transformer mask
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data = data.long().to(device)
            target = target.to(device)

            pred = model(data)

            if model_type == "discriminator":
                loss = criterion(pred.view((-1,)), target.float().view((-1,)))

                # get the accuracy
                prob = torch.sigmoid(pred).view(-1)

                correct += torch.sum(torch.eq(prob > 0.5, target.to(bool)))
                total += len(target)
            else:
                loss = criterion(pred, target.long().view((-1,)))

                # get the accuracy
                prediction = torch.topk(pred, k=1, dim=-1).indices.view(-1)
                valid_idx = target.view(-1) != 0

                correct += torch.sum(torch.eq(prediction[valid_idx], target.view(-1)[valid_idx]))
                total += len(target.view(-1)[valid_idx])

            total_val_loss += loss.item()

    # loss
    val_loss = total_val_loss / len(data_loader)

    if config.verbose and is_main_process():
        print("Validation loss = {:.5f}\t Accuracy = {:.2f}".format(val_loss, correct * 100 / total))

    return {"val_loss": val_loss}


def adv_training(discriminator, generator, config, world_size, device, all_locs, log_dir, input_data):
    train_data, train_idx, vali_data, vali_idx = input_data

    rollout = Rollout(generator, 0.8)

    # gan loss and optimizer
    gen_gan_loss = GANLoss().to(device)
    gen_gan_optm = optim.Adam(generator.parameters(), lr=config.g_lr, weight_decay=config.weight_decay)
    # period loss
    period_crit = periodLoss(time_interval=24).to(device)
    # distance loss
    distance_crit = distanceLoss(locations=all_locs, device=device).to(device)

    # evaluation
    metrics = Metric(config, locations=all_locs, input_data=vali_data, valid_start_end_idx=vali_idx)

    d_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.d_lr, weight_decay=config.weight_decay)

    for epoch in range(config.adv_max_epoch):
        # Train the generator for one step

        if is_main_process():
            print("=" * 50)
            print("training generator")

        set_requires_grad(discriminator, False)
        discriminator.eval()
        generator.train()
        # only once and update rollout parameter
        for _ in range(config.g_step):
            # evaluate current generator performance
            samples = generate_samples(generator, config.generate_len + 1, single_len=64, num=64)
            jsds = metrics.get_individual_jsds(gene_data=samples)

            if is_main_process():
                print(
                    "Metric: distance {:.4f}, rg {:.4f}, period {:.4f}, all_topk {:.4f}, topk {:.4f}".format(
                        jsds[0], jsds[1], jsds[2], jsds[3], jsds[4]
                    )
                )

            # train
            start_time = time.time()
            train_loss = train_generator(
                generator,
                discriminator,
                samples,
                rollout,
                gen_gan_loss,
                gen_gan_optm,
                config,
                device,
                crit=(period_crit, distance_crit),
            )

            if config.verbose and is_main_process():
                print(
                    "Generator: Epoch {}\t loss: {:.3f}, took: {:.2f}s \r".format(
                        epoch + 1,
                        train_loss,
                        time.time() - start_time,
                    )
                )
        rollout.update_params()

        # train discriminator
        if is_main_process():
            print("training discriminator")
        set_requires_grad(discriminator, True)
        discriminator.train()
        generator.eval()
        for _ in range(config.d_step):
            samples = generate_samples(
                generator, config.generate_len, num=config.num_gen_samples, single_len=1024, print_progress=False
            )
            # sample approapriate amount of training data
            curr_train_idx = train_idx
            if len(train_idx) > config.num_gen_samples:
                curr_train_idx = random.sample(train_idx, config.num_gen_samples)
            d_train_data = discriminator_dataset(
                true_data=train_data, fake_data=samples, valid_start_end_idx=curr_train_idx
            )
            train_sampler = DistributedSampler(d_train_data, num_replicas=world_size, rank=device, shuffle=True)
            d_train_loader = torch.utils.data.DataLoader(
                d_train_data,
                collate_fn=discriminator_collate_fn,
                num_workers=config.num_workers,
                batch_size=config.d_batch_size,
                pin_memory=True,
                sampler=train_sampler,
            )
            if is_main_process():
                print(f"len d_train loader:\t{len(d_train_loader)}\t #samples: {len(d_train_data)}")

            for i in range(config.k_d):
                d_train_loader.sampler.set_epoch(i)
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

        # save models
        torch.save(generator.state_dict(), log_dir + "/generator.pt")
        torch.save(discriminator.state_dict(), log_dir + "/discriminator.pt")

        # if epoch > 10:
        #     samples = generate_samples(
        #         generator, config.generate_len, num=config.num_gen_samples, single_len=1024, print_progress=False
        #     )
        #     save_path = os.path.join(config.temp_save_root, "temp", f"generated_samples_{epoch}.pk")
        #     save_pk_file(save_path, samples)


def train_generator(generator, discriminator, samples, rollout, gen_gan_loss, gen_gan_optm, config, device, crit=None):
    period_crit, distance_crit = crit

    # construct the input to the generator, add zeros before samples and delete the last column
    # zeros = torch.zeros((config.d_batch_size, 1)).long().to(device)
    samples = torch.Tensor(samples).long().to(device)

    inputs = samples[:, :-1]
    targets = samples[:, 1:].reshape((-1,))

    # calculate the reward
    rewards = rollout.get_reward(
        samples[:, :-1], roll_out_num=config.rollout_num, discriminator=discriminator, device=device
    )

    prob = generator(inputs)
    prob = F.log_softmax(prob, dim=-1)

    gloss = gen_gan_loss(prob, targets, rewards, device)

    print(gloss.item())

    # additional losses
    # gloss += config.periodic_loss * period_crit(samples)
    gloss += config.distance_loss * distance_crit(samples)

    running_loss = gloss.item()
    # optimize
    gen_gan_optm.zero_grad()
    gloss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
    gen_gan_optm.step()

    return running_loss


def save_pk_file(save_path, data):
    """Function to save data to pickle format given data and path."""
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def set_requires_grad(net, requires_grad=False):
    """Set requies_grad=False for network to avoid unnecessary computations
    Parameters:
        nets (network)   -- a network
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad


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


def is_main_process():
    return get_rank() == 0
