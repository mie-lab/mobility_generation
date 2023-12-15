# coding: utf-8
import pdb
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""

    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward, device):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        one_hot = torch.zeros((target.size(0), prob.size(1)), dtype=torch.bool).to(device)
        one_hot.scatter_(1, target.view((-1, 1)), 1)
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        return -torch.sum(loss)


class distanceLoss(nn.Module):
    def __init__(self, locations, device):
        super(distanceLoss, self).__init__()

        geo_x = locations.sort_values(by="loc_id")["geometry"].x.values
        geo_y = locations.sort_values(by="loc_id")["geometry"].y.values

        self.geo_x = torch.Tensor(geo_x).float().to(device)
        self.geo_y = torch.Tensor(geo_y).float().to(device)

    def forward(self, x):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """

        x = x.long() - 1  # for padding
        x1 = torch.index_select(self.geo_x, 0, x[:, :-1].reshape(-1))
        x2 = torch.index_select(self.geo_x, 0, x[:, 1:].reshape(-1))
        y1 = torch.index_select(self.geo_y, 0, x[:, :-1].reshape(-1))
        y2 = torch.index_select(self.geo_y, 0, x[:, 1:].reshape(-1))

        dx = x1 - x2
        dy = y1 - y2
        loss = dx**2 + dy**2
        loss = torch.mean(loss) / 1000000
        return loss


class periodLoss(nn.Module):
    def __init__(self, time_interval):
        super(periodLoss, self).__init__()
        self.time_interval = time_interval
        self.mse = nn.MSELoss()

    def forward(self, x):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """
        # TODO: change to true periodic loss for 24h
        loss = 0.0
        for i in range(x.size(1) - self.time_interval):
            loss += torch.sum(torch.ne(x[:, i], x[:, i + self.time_interval]))

        loss = loss / (x.size(0) * (x.size(1) - self.time_interval))
        return loss
