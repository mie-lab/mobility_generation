# coding: utf-8
import pdb
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


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
        one_hot = torch.zeros((target.size(0), prob.size(1))).to(device)
        one_hot.scatter_(1, target.view((-1, 1)), 1)
        one_hot = one_hot.type(torch.BoolTensor)
        one_hot = Variable(one_hot).to(device)

        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        return -torch.mean(loss)


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
        x = x.long()
        x1 = torch.index_select(self.geo_x, 0, x[:, :-1].reshape(-1))
        x2 = torch.index_select(self.geo_x, 0, x[:, 1:].reshape(-1))
        y1 = torch.index_select(self.geo_y, 0, x[:, :-1].reshape(-1))
        y2 = torch.index_select(self.geo_y, 0, x[:, 1:].reshape(-1))

        dx = x1 - x2
        dy = y1 - y2
        loss = dx**2 + dy**2
        loss = torch.sum(loss) / loss.size(0)
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
        loss = 0.0
        for i in range(x.size(1) - self.time_interval):
            loss += torch.sum(torch.ne(x[:, i], x[:, i + self.time_interval]))

        loss = loss / (x.size(0) * (x.size(1) - self.time_interval))
        return loss


class embd_distance_loss(nn.Module):
    def __init__(self, embd):
        super(embd_distance_loss, self).__init__()
        self.embd = embd

    def forward(self, x, embd_size):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """
        emb = self.embd(x)
        emb = emb.permute(1, 0, 2)
        curr = emb[: x.size(1) - 1].contiguous().view(-1, embd_size)
        next = emb[1 : x.size(1)].contiguous().view(-1, embd_size)
        loss = torch.nn.functional.mse_loss(curr, next, reduction="sum")
        return loss


class embd_period_loss(nn.Module):
    def __init__(self, embd):
        super(embd_period_loss, self).__init__()
        self.embd = embd

    def forward(self, x, embd_size):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """
        emb = self.embd(x)
        emb = emb.permute(1, 0, 2)
        curr = emb[:24].contiguous().view(-1, embd_size)
        next = emb[24:].contiguous().view(-1, embd_size)
        loss = torch.nn.functional.mse_loss(curr, next, reduction="sum")
        return loss
