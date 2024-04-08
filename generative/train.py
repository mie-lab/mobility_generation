import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import torch.distributed as dist

import os
import random
import numpy as np
from tqdm import tqdm

import pickle as pickle

import time

from generative.dataloader import get_discriminator_dataloaders, get_pretrain_loaders, generate_samples
from generative.rollout import Rollout
from generative.gan_loss import GANLoss, periodLoss, distanceLoss


from metrics.evaluations import Metric
from utils.earlystopping import EarlyStopping

from utils.dist_util import load_state_dict
import blobfile as bf


def send_to_device(inputs, device, model_type="discriminator"):
    if model_type == "discriminator":
        x, y, x_dict = inputs

        x = x.to(device)
        for key in x_dict:
            x_dict[key] = x_dict[key].to(device)
        y = y.to(device)
        return x, y, x_dict
    else:
        x, y, x_dict, y_dict = inputs

        x = x.to(device)
        for key in x_dict:
            x_dict[key] = x_dict[key].to(device)
        for key in y_dict:
            y_dict[key] = y_dict[key].to(device)
        y = y.to(device)
        return x, y, x_dict, y_dict


def pre_training(discriminator, generator, config, world_size, device, log_dir, input_data):
    d_train_loader, d_vali_loader, g_train_loader, g_vali_loader = get_pretrain_loaders(
        input_data, world_size, config, device
    )

    # loss and optimizer
    d_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(device)
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=config.pre_lr, weight_decay=config.weight_decay, eps=1e-5)
    g_criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=0).to(device)  # ignore_index for padding
    g_optimizer = optim.AdamW(generator.parameters(), lr=config.pre_lr, weight_decay=config.weight_decay, eps=1e-5)

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
        delta=0.0001,
        monitor="val_loss",
    )

    # Time for printing
    training_start_time = time.time()
    scheduler_count = 0

    scaler = torch.cuda.amp.GradScaler(enabled=config.use_fp16)

    # Loop for n_epochs
    for epoch in range(config.max_epoch):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(config, model, train_loader, optimizer, criterion, device, epoch, model_type, scaler)

        # At the end of the epoch, do a pass on the validation set
        return_dict = validate_epoch(config, model, val_loader, criterion, device, model_type)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "lr_schedule": scheduler.state_dict(),
        }

        early_stopping(return_dict, checkpoint, save_name=f"{model_type}")

        if early_stopping.early_stop:
            if config.verbose and is_main_process():
                print("=" * 50)
                print("Early stopping")

            if scheduler_count == 2:
                # early_stopping.best_return_dict
                if is_main_process():
                    print("Training finished.\t Time: {:.2f}s.".format((time.time() - training_start_time)))
                break

            # for multigpu
            dist.barrier()

            checkpoint = load_state_dict(bf.join(log_dir, early_stopping.save_name + ".pt"))
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            scheduler.load_state_dict(checkpoint["lr_schedule"])

            #
            scheduler_count += 1
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler.step()
            if config.verbose and is_main_process():
                print("Current learning rate: {:.6f}".format(optimizer.param_groups[0]["lr"]))

        if config.verbose:
            print("=" * 50)
        if config.debug is True:
            break

    return model


def train_epoch(config, model, data_loader, optimizer, criterion, device, epoch, model_type, scaler):
    n_batches = len(data_loader)

    model.train()

    running_loss = 0.0
    training_loss = 0
    all_correct = 0
    all_total = 0
    curr_correct = 0
    curr_total = 0
    MSE = torch.nn.MSELoss(reduction="mean")

    start_time = time.time()

    for i, inputs in enumerate(data_loader):
        optimizer.zero_grad()

        if model_type == "discriminator":
            x, y, x_dict = send_to_device(inputs, device, model_type=model_type)
            x = x.long()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.use_fp16):
                pred = model(x, x_dict)
                loss = criterion(pred.view((-1,)), y.float().view((-1,)))

            # get the accuracy
            prob = torch.sigmoid(pred).view(-1)

            correct = torch.sum(torch.eq(prob > 0.5, y.to(bool)))
            total = len(y)

            all_correct += correct
            all_total += total
            curr_correct += correct
            curr_total += total

        else:
            x, y, x_dict, y_dict = send_to_device(inputs, device, config)
            x = x.long()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.use_fp16):
                loc_pred, dur_pred = model(x, x_dict)

                loc_loss_size = criterion(loc_pred, y.long().view((-1,)))
                dur_loss_size = MSE(dur_pred.reshape(-1), y_dict["duration"].reshape(-1))
                loss = loc_loss_size + config.loss_weight * dur_loss_size / (dur_loss_size / loc_loss_size).detach()

            # get the accuracy
            prediction = torch.topk(loc_pred, k=1, dim=-1).indices.view(-1)
            valid_idx = y.view(-1) != 0

            correct = torch.sum(torch.eq(prediction[valid_idx], y.view(-1)[valid_idx]))
            total = len(y.view(-1)[valid_idx])

            all_correct += correct
            all_total += total
            curr_correct += correct
            curr_total += total

        if torch.isnan(loss):
            assert False
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler.step(optimizer)
        scaler.update()

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

        if config.debug and (i > 20):
            break

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

    MSE = torch.nn.MSELoss(reduction="mean")

    # change to validation mode. Warning in inference mode regarding the transformer mask
    model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            if model_type == "discriminator":
                x, y, x_dict = send_to_device(inputs, device, model_type=model_type)
                x = x.long()

                pred = model(x, x_dict)

                loss = criterion(pred.view((-1,)), y.float().view((-1,)))

                # get the accuracy
                prob = torch.sigmoid(pred).view(-1)

                correct += torch.sum(torch.eq(prob > 0.5, y.to(bool)))
                total += len(y)
            else:
                x, y, x_dict, y_dict = send_to_device(inputs, device, config)
                x = x.long()

                loc_pred, dur_pred = model(x, x_dict)

                loc_loss_size = criterion(loc_pred, y.long().view((-1,)))
                dur_loss_size = MSE(dur_pred.reshape(-1), y_dict["duration"].reshape(-1))
                loss = loc_loss_size + config.loss_weight * dur_loss_size / (dur_loss_size / loc_loss_size).detach()

                # get the accuracy
                prediction = torch.topk(loc_pred, k=1, dim=-1).indices.view(-1)
                valid_idx = y.view(-1) != 0

                correct += torch.sum(torch.eq(prediction[valid_idx], y.view(-1)[valid_idx]))
                total += len(y.view(-1)[valid_idx])

            total_val_loss += loss.item()

            if config.debug:
                break

    # loss
    val_loss = total_val_loss / len(data_loader)

    if config.verbose and is_main_process():
        print("Validation loss = {:.5f}\t Accuracy = {:.2f}".format(val_loss, correct * 100 / total))

    return {"val_loss": val_loss}


def adv_training(discriminator, generator, config, world_size, device, all_locs, log_dir, input_data):
    train_data, vali_data = input_data

    # get the data
    true_seqs = {"locs": [], "durs": []}
    for x, _, x_dict, _ in tqdm(train_data):
        x = x.squeeze().numpy().copy()
        duration = x_dict["duration"].squeeze().numpy().copy()
        try:
            np.arange(len(x))
        except TypeError:
            continue

        true_seqs["locs"].append(x)
        true_seqs["durs"].append(duration)

    rollout = Rollout(generator, 0.8)

    scaler = torch.cuda.amp.GradScaler(enabled=config.use_fp16)

    # gan loss and optimizer
    gen_gan_loss = GANLoss().to(device)
    gen_gan_optm = optim.AdamW(generator.parameters(), lr=config.g_lr, weight_decay=config.weight_decay)
    # period loss
    period_crit = periodLoss(time_interval=24).to(device)
    # distance loss
    distance_crit = distanceLoss(locations=all_locs, device=device).to(device)

    # evaluation
    # metrics = Metric(config, locations=all_locs, input_data=vali_data, valid_start_end_idx=vali_idx)

    d_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(device)
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=config.d_lr, weight_decay=config.weight_decay)

    for epoch in range(config.max_epoch):
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
            samples = generate_samples(generator, config.generate_len + 1, single_len=64, num=128)
            # jsds = metrics.get_individual_jsds(gene_data=samples)

            # if is_main_process():
            #     print(
            #         "Metric: distance {:.4f}, rg {:.4f}, period {:.4f}, all_topk {:.4f}, topk {:.4f}".format(
            #             jsds[0], jsds[1], jsds[2], jsds[3], jsds[4]
            #         )
            #     )

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
                generator, config.generate_len, num=config.num_gen_samples, single_len=1024, print_progress=True
            )
            # sample approapriate amount of training data
            curr_train_idx = np.arange(len(true_seqs["locs"]))
            if len(curr_train_idx) > config.num_gen_samples:
                curr_train_idx = np.array(random.sample(range(len(curr_train_idx)), config.num_gen_samples))
                curr_train_idx = np.sort(curr_train_idx).astype(int)

            d_train_data, d_train_loader = get_discriminator_dataloaders(
                train_data={
                    "locs": [true_seqs["locs"][i] for i in curr_train_idx],
                    "durs": [true_seqs["durs"][i] for i in curr_train_idx],
                },
                fake_data=samples,
                world_size=world_size,
                config=config,
                device=device,
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
                    scaler=scaler,
                )

        if epoch > 1:
            samples = generate_samples(
                generator,
                config.generate_len + 1,
                num=config.num_gen_samples,
                single_len=1024,
                print_progress=False,
            )
            save_path = os.path.join("runs", "temp", f"generated_samples_{epoch}.pk")
            with open(save_path, "wb") as handle:
                pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save models
        torch.save(generator.state_dict(), log_dir + "/generator_final.pt")
        torch.save(discriminator.state_dict(), log_dir + "/discriminator_final.pt")


def train_generator(generator, discriminator, samples, rollout, gen_gan_loss, gen_gan_optm, config, device, crit=None):
    period_crit, distance_crit = crit

    MSE = torch.nn.MSELoss(reduction="mean")

    # construct the input to the generator, add zeros before samples and delete the last column
    # zeros = torch.zeros((config.batch_size, 1)).long().to(device)
    locs = torch.Tensor(samples["locs"]).long().to(device)
    durs = torch.Tensor(samples["durs"]).long().to(device)

    x = locs[:, :-1]
    targets = locs[:, 1:].reshape((-1,))
    x_dict = {"duration": durs[:, :-1]}

    # calculate the reward
    rewards = rollout.get_reward(x, x_dict, roll_out_num=config.rollout_num, discriminator=discriminator, device=device)

    prob, dur_pred = generator(x, x_dict)
    prob = F.log_softmax(prob, dim=-1)

    dur_loss = MSE(dur_pred.reshape(-1), x_dict["duration"].reshape(-1).float())

    gloss = gen_gan_loss(prob, dur_loss, targets, rewards, device, loss_weight=config.loss_weight)

    # additional losses
    distance_loss = distance_crit(locs)
    # gloss += config.loss_weight * period_crit(samples)

    gloss = gloss + config.loss_weight * distance_loss / (distance_loss / gloss).detach()

    running_loss = gloss.item()
    # optimize
    gen_gan_optm.zero_grad()
    gloss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
    gen_gan_optm.step()

    return running_loss


def set_requires_grad(net, requires_grad=False):
    """Set requies_grad=False for network to avoid unnecessary computations
    Parameters:
        nets (network)   -- a network
        requires_grad (bool)  -- whether the networks require gradients or not
    """
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
