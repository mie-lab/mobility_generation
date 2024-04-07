import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import math


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return token_embedding + self.pos_embedding[: token_embedding.size(0), :]


class TemporalEmbedding(nn.Module):
    def __init__(self, d_input, emb_info="all"):
        super(TemporalEmbedding, self).__init__()

        self.emb_info = emb_info
        self.minute_size = 4
        hour_size = 24
        weekday = 7

        if self.emb_info == "all":
            # add 1 for padding
            self.minute_embed = nn.Embedding(self.minute_size + 1, d_input)
            self.hour_embed = nn.Embedding(hour_size + 1, d_input)
            self.weekday_embed = nn.Embedding(weekday + 1, d_input)
        elif self.emb_info == "time":
            self.time_embed = nn.Embedding(self.minute_size * hour_size, d_input)
        elif self.emb_info == "weekday":
            self.weekday_embed = nn.Embedding(weekday, d_input)

    def forward(self, time, weekday):
        if self.emb_info == "all":
            hour = torch.div(time, self.minute_size, rounding_mode="floor")
            minutes = time % 4

            minute_x = self.minute_embed(minutes)
            hour_x = self.hour_embed(hour)
            weekday_x = self.weekday_embed(weekday)

            return hour_x + minute_x + weekday_x
        elif self.emb_info == "time":
            return self.time_embed(time)
        elif self.emb_info == "weekday":
            return self.weekday_embed(weekday)


class POINet(nn.Module):
    def __init__(self, poi_vector_size, out):
        super(POINet, self).__init__()

        # poi_vector_size -> poi_vector_size*4 -> poi_vector_size
        self.linear1 = torch.nn.Linear(poi_vector_size, poi_vector_size * 4)
        self.linear2 = torch.nn.Linear(poi_vector_size * 4, poi_vector_size)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.norm2 = nn.LayerNorm(poi_vector_size)

        # poi_vector_size -> out
        self.fc = nn.Linear(poi_vector_size, out)

    def forward(self, x):
        x = self.norm2(x + self._dense_block(x))
        return self.fc(x)

    def _dense_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout1(F.relu(self.linear1(x))))
        return self.dropout2(x)


class AllEmbedding(nn.Module):
    def __init__(self, d_input, config, if_pos_encoder=True, emb_info="all") -> None:
        super(AllEmbedding, self).__init__()
        # emberdding layers
        self.d_input = d_input

        # location embedding
        self.emb_loc = nn.Embedding(config.max_location, d_input)

        # time is in minutes, possible time for each day is 60 * 24 // 15
        self.if_include_time = config.if_embed_time
        if self.if_include_time:
            self.temporal_embedding = TemporalEmbedding(d_input, emb_info)

        # duration is in minutes, possible duration for two days is 60 * 24 * 2 // 30
        self.if_include_duration = config.if_embed_duration
        if self.if_include_duration:
            # add 1 for padding
            self.emb_duration = nn.Embedding((60 * 24 * 2) // 30 + 1, d_input)

        # poi
        self.if_include_poi = config.if_embed_poi
        if self.if_include_poi:
            self.poi_net = POINet(config.poi_original_size, d_input)

        # position encoder for transformer
        self.if_pos_encoder = if_pos_encoder
        if self.if_pos_encoder:
            self.pos_encoder = PositionalEncoding(d_input, dropout=0.1)

    def forward(self, src, context_dict) -> Tensor:
        emb = self.emb_loc(src)

        if self.if_include_time:
            emb = emb + self.temporal_embedding(context_dict["time"], context_dict["weekday"])

        if self.if_include_duration:
            emb = emb + self.emb_duration(context_dict["duration"])

        if self.if_include_poi:
            emb = emb + self.poi_net(context_dict["poi"])

        if self.if_pos_encoder:
            return self.pos_encoder(emb * math.sqrt(self.d_input))
        else:
            return emb
