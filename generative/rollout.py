# -*- coding:utf-8 -*-

import copy
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm


class Rollout(object):
    """Roll-out policy"""

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x, roll_out_num, discriminator, device):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        rewards = torch.zeros((batch_size, seq_len)).to(device)

        for _ in range(roll_out_num):
            for step in range(1, seq_len):
                data = x[:, :step]
                # use own model
                samples = self.own_model.sample(batch_size, seq_len, data)
                pred = discriminator(samples)

                rewards[:, step - 1] += F.sigmoid(pred).detach().view((-1,))

            # for the last token
            pred = discriminator(x)
            rewards[:, seq_len - 1] += F.sigmoid(pred).detach().view((-1,))

        rewards = rewards / roll_out_num
        return rewards.view((-1,))  # batch_size * seq_len

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith("emb") or name.startswith("Emb"):
                param.data = dic[name]
            else:
                # update own model
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
