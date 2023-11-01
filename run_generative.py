# coding=utf-8
import pdb
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm

from torch import nn, optim

from train import generate_samples, pretrain_model, train_epoch
from utils import get_workspace_logger, read_data_from_file
from rollout import Rollout
from evaluations import IndividualEval
from gen_data import gen_matrix
from models.generator import ATGenerator
from models.discriminator import Discriminator
from models.gan_loss import GANLoss, distance_loss, period_loss
from data_iter import GenDataIter, NewGenIter, DisDataIter


def main(opt):
    # all parameters
    # assigned in argparse
    print(opt)

    # fixed parameters
    SEED = 88
    EPOCHS = 30
    BATCH_SIZE = 128
    SEQ_LEN = 48
    GENERATED_NUM = 10000

    DATA_PATH = "./data"
    REAL_DATA = DATA_PATH + "/geolife/real.data"
    VAL_DATA = DATA_PATH + "/geolife/val.data"
    TEST_DATA = DATA_PATH + "/geolife/test.data"
    GENE_DATA = DATA_PATH + "/geolife/gene.data"

    random.seed(SEED)
    np.random.seed(SEED)

    TOTAL_LOCS = 23768
    individualEval = IndividualEval(data="geolife")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Pre-processing Data...")
    gen_matrix()

    # assigned according to task
    if opt.task == "attention":
        d_pre_epoch = 20
        g_pre_epoch = 110
        ploss_alpha = float(opt.ploss)
        dloss_alpha = float(opt.dloss)
        generator = ATGenerator(
            device=device,
            total_locations=TOTAL_LOCS,
            starting_sample="real",
            starting_dist=np.load(f"{DATA_PATH}/geolife/start.npy"),
        )
        discriminator = Discriminator(total_locations=TOTAL_LOCS)
        gen_train_fixstart = True
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # prepare files and datas
    logger = get_workspace_logger()

    if opt.pretrain:
        generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM, DATA_PATH + "/geolife/gene_epoch_init.data")

        # pretrain discriminator
        logger.info("pretrain discriminator ...")
        pretrain_real = DATA_PATH + "/geolife/real.data"
        pretrain_fake = DATA_PATH + "/geolife/dispre.data"
        dis_data_iter = DisDataIter(pretrain_real, pretrain_fake, BATCH_SIZE, SEQ_LEN)
        dis_criterion = nn.NLLLoss(reduction="mean").to(device)
        dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.000001)

        pretrain_model(
            "D", d_pre_epoch, discriminator, dis_data_iter, dis_criterion, dis_optimizer, BATCH_SIZE, device=device
        )

        # pretrain generator
        logger.info("pretrain generator ...")
        if gen_train_fixstart:
            gen_data_iter = NewGenIter(REAL_DATA, BATCH_SIZE)
        else:
            gen_data_iter = GenDataIter(REAL_DATA, BATCH_SIZE)
        gen_criterion = nn.NLLLoss(reduction="mean")
        gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
        gen_criterion = gen_criterion.to(device)
        pretrain_model(
            "G", g_pre_epoch, generator, gen_data_iter, gen_criterion, gen_optimizer, BATCH_SIZE, device=device
        )
        torch.save(generator.state_dict(), DATA_PATH + "/geolife/pretrain/generator.pth")
        torch.save(discriminator.state_dict(), DATA_PATH + "/geolife/pretrain/discriminator.pth")

    else:
        generator.load_state_dict(torch.load(DATA_PATH + "/geolife/pretrain/generator.pth"))
        discriminator.load_state_dict(torch.load(DATA_PATH + "/geolife/pretrain/discriminator.pth"))
        print("")
    print("advtrain generator and discriminator ...")
    rollout = Rollout(generator, 0.8)

    gen_gan_loss = GANLoss().to(device)
    gen_gan_optm = optim.Adam(generator.parameters(), lr=0.0001)

    dis_criterion = nn.NLLLoss(reduction="mean").to(device)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)

    generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM, GENE_DATA)
    generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM, DATA_PATH + "/geolife/gene_epoch_0.data")

    for epoch in range(EPOCHS):
        gene_data = read_data_from_file(GENE_DATA)
        val_data = read_data_from_file(VAL_DATA)

        # evaluation and save results
        JSDs = individualEval.get_individual_jsds(t1=gene_data, t2=val_data)

        with open(DATA_PATH + "/geolife/logs/jsd.log", "a") as f:
            f.write(" ".join([str(j) for j in JSDs]))
            f.write("\n")

        print("Current JSD: %f, %f, %f, %f, %f, %f" % (JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]))

        # Train the generator for one step
        for it in range(1):
            samples = generator.sample(BATCH_SIZE, SEQ_LEN)
            # construct the input to the generator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor).to(device)

            inputs = torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous()
            tim = torch.LongTensor([i % 24 for i in range(48)]).to(device)
            tim = tim.repeat(BATCH_SIZE).reshape(BATCH_SIZE, -1)

            targets = samples.contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 16, discriminator)
            rewards = torch.Tensor(rewards).to(device)
            rewards = torch.exp(rewards).contiguous().view((-1,))
            prob = generator.forward(inputs, tim)

            gloss = gen_gan_loss(prob, targets, rewards, device)

            # additional losses
            if ploss_alpha != 0.0:
                p_crit = period_loss(24).to(device)
                gloss += ploss_alpha * p_crit(samples.float())

            if dloss_alpha != 0.0:
                d_crit = distance_loss(device=device).to(device)
                gloss += dloss_alpha * d_crit(samples.float())

            # optimize
            gen_gan_optm.zero_grad()
            gloss.backward()
            gen_gan_optm.step()

        rollout.update_params()

        print("training discriminator")
        for _ in tqdm(range(4)):
            generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM, GENE_DATA)
            dis_data_iter = DisDataIter(REAL_DATA, GENE_DATA, BATCH_SIZE, SEQ_LEN)
            for _ in range(2):
                dloss = train_epoch(
                    "D", discriminator, dis_data_iter, dis_criterion, dis_optimizer, BATCH_SIZE, device=device
                )

        logger.info("Epoch [%d] Generator Loss: %f, Discriminator Loss: %f" % (epoch, gloss.item(), dloss))
        with open(DATA_PATH + "/geolife/logs/loss.log", "a") as f:
            f.write(" ".join([str(j) for j in [epoch, float(gloss.item()), dloss]]))
            f.write("\n")
        if (epoch + 1) % 20 == 0:
            generate_samples(
                generator,
                BATCH_SIZE,
                SEQ_LEN,
                GENERATED_NUM,
                DATA_PATH + "/geolife/gene_epoch_%d.data" % (epoch + 1),
            )

    # evaluation
    test_data = read_data_from_file(TEST_DATA)
    gene_data = read_data_from_file(GENE_DATA)
    JSDs = individualEval.get_individual_jsds(t1=gene_data, t2=test_data)
    print("Test JSD: %f, %f, %f, %f, %f, %f" % (JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]))

    torch.save(generator.state_dict(), DATA_PATH + "/geolife/generator.pth")
    torch.save(discriminator.state_dict(), DATA_PATH + "/geolife/discriminator.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--task", default="attention", type=str)
    parser.add_argument("--ploss", default="3.0", type=float)
    parser.add_argument("--dloss", default="1.5", type=float)
    parser.add_argument("--length", default=48, type=int)

    opt = parser.parse_args()
    main(opt)
