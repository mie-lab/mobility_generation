import torch.nn as nn
import torch
from torch import Tensor

import torch.nn.functional as F


class FullyConnected(nn.Module):
    def __init__(self, d_input, config, if_residual_layer=True):
        super(FullyConnected, self).__init__()
        # the last fully connected layer
        fc_dim = d_input

        self.fc_loc = nn.Linear(fc_dim, config.total_loc_num)
        self.emb_dropout = nn.Dropout(p=0.1)

        self.if_residual_layer = if_residual_layer
        if self.if_residual_layer:
            # the residual
            self.fc_1 = nn.Linear(fc_dim, fc_dim)
            self.norm_1 = nn.BatchNorm1d(fc_dim)
            self.fc_dropout = nn.Dropout(p=config.fc_dropout)

    def forward(self, out) -> Tensor:
        # with fc output
        out = self.emb_dropout(out)

        # residual
        if self.if_residual_layer:
            out = self.norm_1(out + self.fc_dropout(F.relu(self.fc_1(out))))

        return self.fc_loc(out)
