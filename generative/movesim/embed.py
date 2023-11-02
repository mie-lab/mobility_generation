import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import math


class AllEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super(AllEmbedding, self).__init__()
        # emberdding layers

        # location embedding
        self.emb_loc = nn.Embedding(num_embeddings=config.total_loc_num, embedding_dim=config.base_emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src) -> Tensor:
        emb = self.emb_loc(src)

        return self.dropout(emb)
