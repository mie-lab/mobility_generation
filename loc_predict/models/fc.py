import torch.nn as nn
import torch
from torch import Tensor

import torch.nn.functional as F


class FullyConnected(nn.Module):
    def __init__(self, d_input, config, if_residual_layer=True):
        super(FullyConnected, self).__init__()
        # the last fully connected layer
        fc_dim = d_input

        self.fc_loc = nn.Linear(fc_dim, config.max_location)
        # self.fc_dur = nn.Linear(fc_dim, 1)
        self.emb_dropout = nn.Dropout(p=0.1)

        self.if_residual_layer = if_residual_layer
        if self.if_residual_layer:
            # the residual
            self.linear1 = nn.Linear(fc_dim, fc_dim * 4)
            self.linear2 = nn.Linear(fc_dim * 4, fc_dim)

            self.norm1 = nn.BatchNorm1d(fc_dim)
            self.fc_dropout1 = nn.Dropout(p=config.fc_dropout)
            self.fc_dropout2 = nn.Dropout(p=config.fc_dropout)

        # the residual
        # self.linear3 = nn.Linear(fc_dim, fc_dim * 2)
        # self.linear4 = nn.Linear(fc_dim * 2, fc_dim)

        # self.norm2 = nn.BatchNorm1d(fc_dim)
        # self.fc_dropout3 = nn.Dropout(p=config.fc_dropout)
        # self.fc_dropout4 = nn.Dropout(p=config.fc_dropout)

    def forward(self, out, user) -> Tensor:
        # with fc output
        out = self.emb_dropout(out)

        # residual
        if self.if_residual_layer:
            out = self.norm1(out + self._res_block(out))

        # return self.fc_loc(out), self.fc_dur(self.norm2(out + self._res_block_dur(out)))
        return self.fc_loc(out)

    def _res_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.fc_dropout1(F.relu(self.linear1(x))))
        return self.fc_dropout2(x)

    def _res_block_dur(self, x: Tensor) -> Tensor:
        x = self.linear4(self.fc_dropout3(F.relu(self.linear3(x))))
        return self.fc_dropout4(x)
