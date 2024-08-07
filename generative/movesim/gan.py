# coding: utf-8
import pickle as pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AllEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super(AllEmbedding, self).__init__()
        # emberdding layers

        # location embedding
        self.emb_loc = nn.Embedding(config.max_location, config.base_emb_size, padding_idx=0)
        # duration is in minutes, possible duration for two days is 60 * 24 * 2 // 30
        self.if_include_duration = config.if_include_duration
        if self.if_include_duration:
            # add 1 for padding
            self.emb_duration = nn.Embedding((60 * 24 * 2) // 30 + 1, config.base_emb_size, padding_idx=0)

        self.dropout = nn.Dropout(0.1)

    def forward(self, src, context_dict) -> Tensor:
        emb = self.emb_loc(src)

        if self.if_include_duration:
            emb = emb + self.emb_duration(context_dict["duration"])

        return self.dropout(emb)


class Discriminator(nn.Module):
    """Basic discriminator."""

    def __init__(self, config, dropout=0.5):
        super(Discriminator, self).__init__()
        self.num_filters = [32, 32, 64, 64, 32]
        self.kernel_sizes = [3, 5, 7, 9, 3]

        self.embedding = AllEmbedding(config=config)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n, (f, config.base_emb_size)) for (n, f) in zip(self.num_filters, self.kernel_sizes)]
        )
        self.highway = nn.Linear(sum(self.num_filters), sum(self.num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sum(self.num_filters), 1)

        self.init_parameters()

    def forward(self, x, context_dict):
        """
        Args:
            x: (batch_size * seq_len)
        """
        padding_mask = x != 0  # batch_size * seq_len

        emb = self.embedding(x, context_dict).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim

        convs = [
            F.relu(conv(emb)).squeeze(3)  # batch_size * num_filter * seq_len
            * padding_mask.unsqueeze(1).repeat(1, n, 1)[..., (k // 2) : -(k // 2)]  # for padding
            for (conv, n, k) in zip(self.convs, self.num_filters, self.kernel_sizes)
        ]

        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1.0 - torch.sigmoid(highway)) * pred

        return self.linear(self.dropout(pred))

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


class Generator(nn.Module):
    """Attention Generator."""

    def __init__(
        self,
        config,
        device=None,
        starting_sample="rand",
        dist_matrix=None,
        emp_matrix=None,
        fct_matrix=None,
        starting_dist=None,
    ):
        """

        :param starting_sample:
        :param starting_dist: prior knowledge of location preferences, could be empirical visit frequency. shape of [#locs]
        """
        super(Generator, self).__init__()
        self.base_emb_size = config.base_emb_size
        self.hidden_dim = config.hidden_dim
        self.total_loc_num = config.max_location

        self.device = device
        self.starting_sample = starting_sample

        if self.starting_sample == "real":
            self.starting_dist = torch.tensor(starting_dist, dtype=torch.float)

        self.dist_matrix = dist_matrix
        self.emp_matrix = emp_matrix
        self.fct_matrix = fct_matrix

        self.embedding = AllEmbedding(config=config)
        # self.tim_embedding = nn.Embedding(num_embeddings=24, embedding_dim=self.base_emb_size)

        # transformer encoder
        self.attn = nn.MultiheadAttention(self.hidden_dim, 4, batch_first=True)
        self.Q = nn.Linear(self.base_emb_size, self.hidden_dim)
        self.V = nn.Linear(self.base_emb_size, self.hidden_dim)
        self.K = nn.Linear(self.base_emb_size, self.hidden_dim)

        self.attn2 = nn.MultiheadAttention(self.hidden_dim, 1, batch_first=True)
        self.Q2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # self.fc = FullyConnected(self.base_emb_size, if_residual_layer=True, total_loc_num=self.total_loc_num)
        self.linear = nn.Linear(self.hidden_dim, self.total_loc_num)

        # the residual for duration
        self.fc_dur = nn.Linear(self.hidden_dim, 1)
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.linear2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.norm1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc_dropout1 = nn.Dropout(p=0.1)
        self.fc_dropout2 = nn.Dropout(p=0.1)

        # for distance, empirical, and function matrics
        # self.linear_mat1 = nn.Linear(self.total_loc_num - 1, self.hidden_dim)
        # self.linear_mat2 = nn.Linear(self.total_loc_num - 1, self.hidden_dim)
        # self.linear_mat3 = nn.Linear(self.total_loc_num - 1, self.hidden_dim)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), True, dtype=torch.bool), diagonal=1)

    def matrix_calculation(self, input):
        locs = input.reshape(-1).detach().cpu().numpy() - 1  # for padding
        dist_vec = self.dist_matrix[locs]
        visit_vec = self.emp_matrix[locs]
        fct_vec = self.fct_matrix[locs]
        dist_vec = torch.tensor(dist_vec, dtype=torch.float, device=self.device)
        visit_vec = torch.tensor(visit_vec, dtype=torch.float, device=self.device)
        fct_vec = torch.tensor(fct_vec, dtype=torch.float, device=self.device)

        # matrix calculation
        dist_vec = torch.sigmoid(self.linear_mat1(dist_vec))
        dist_vec = F.normalize(dist_vec)

        visit_vec = torch.sigmoid(self.linear_mat2(visit_vec))
        visit_vec = F.normalize(visit_vec)

        fct_vec = torch.sigmoid(self.linear_mat3(fct_vec))
        fct_vec = F.normalize(fct_vec)

        return dist_vec, visit_vec, fct_vec

    def _res_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.fc_dropout1(F.relu(self.linear1(x))))
        return self.fc_dropout2(x)

    def _single_step(self, input, context_dict):
        src_padding_mask = (input == 0).to(self.device)

        x = self.embedding(input, context_dict)

        #
        src_mask = self._generate_square_subsequent_mask(input.shape[1]).to(self.device)

        Query = F.relu(self.Q(x))
        Value = F.relu(self.V(x))
        Key = F.relu(self.K(x))

        x, _ = self.attn(Query, Key, Value, key_padding_mask=src_padding_mask, attn_mask=src_mask, need_weights=False)

        Query = F.relu(self.Q2(x))
        Value = F.relu(self.V2(x))
        Key = F.relu(self.K2(x))

        x, _ = self.attn2(Query, Key, Value, key_padding_mask=src_padding_mask, attn_mask=src_mask, need_weights=False)

        # for padding
        # out = out * (~src_padding_mask).unsqueeze(-1).repeat(1, 1, self.base_emb_size)
        x = x.reshape(-1, self.hidden_dim)
        # dist_vec, visit_vec, fct_vec = self.matrix_calculation(input)
        # x = x + torch.mul(x, dist_vec) + torch.mul(x, visit_vec) + torch.mul(x, fct_vec)

        return self.linear(x), self.fc_dur(self.norm1(x + self._res_block(x)))

    def forward(self, input, context_dict):
        """
        :param x: (batch_size, seq_len), sequence of locations
        :return:
            (batch_size * seq_len, total_locations), prediction of next stage of all locations
        """
        loc_pred, dur_pred = self._single_step(input, context_dict)

        return loc_pred, dur_pred

    def step(self, input, context_dict):
        """
        :param x: (batch_size, 1), current location
        :return:
            (batch_size, total_locations), prediction of next stage
        """
        loc_pred, dur_pred = self._single_step(input, context_dict)

        return F.softmax(loc_pred, dim=-1), dur_pred

    def sample(self, batch_size, seq_len, x=None, x_dict=None):
        """
        :param batch_size: int, size of a batch of training data
        :param seq_len: int, length of the sequence
        :param x: (batch_size, k), current generated sequence
        :return: (batch_size, seq_len), complete generated sequence
        """
        duration_upper_limit = 60 * 24 * 2 // 30 + 1
        flag = False

        if x is None:
            flag = True

        s = 0
        if flag:
            if self.starting_sample == "rand":
                x = torch.randint(low=1, high=self.total_loc_num, size=(batch_size, 1)).long().to(self.device)
            elif self.starting_sample == "real":
                x = (
                    torch.stack([self.starting_dist.multinomial(1) for _ in range(batch_size)], dim=0)
                    .long()
                    .to(self.device)
                )
            if (x == 0).any():
                assert False  # for padding
            s = 1
            # random sample duration
            duration = torch.randint(low=1, high=duration_upper_limit, size=(batch_size, 1)).long().to(self.device)

        samples = {"locs": [], "durs": []}
        if flag:
            if s > 0:
                samples["locs"].append(x)
                samples["durs"].append(duration)

            for i in range(s, seq_len):
                x, duration = self.step(x, {"duration": duration})
                x = x.multinomial(1)
                x[x == 0] += 1  # for padding

                duration += 1
                duration = torch.clamp(torch.round(duration), min=1, max=duration_upper_limit).long()

                samples["locs"].append(x)
                samples["durs"].append(duration)
        else:
            given_len = x.size(1)

            lis_loc = x.chunk(given_len, dim=1)
            lis_dur = x_dict["duration"].chunk(given_len, dim=1)
            for i in range(given_len):
                samples["locs"].append(lis_loc[i])
                samples["durs"].append(lis_dur[i])

            x, dur_pred = self.step(lis_loc[-1], {"duration": lis_dur[-1]})
            x = x.multinomial(1)
            x[x == 0] += 1  # for padding

            dur_pred += 1
            dur_pred = torch.clamp(torch.round(dur_pred), min=1, max=duration_upper_limit).long()

            for i in range(given_len, seq_len):
                samples["locs"].append(x)
                samples["durs"].append(dur_pred)

                x, dur_pred = self.step(x, {"duration": dur_pred})
                x = x.multinomial(1)
                x[x == 0] += 1  # for padding

                dur_pred += 1
                dur_pred = torch.clamp(torch.round(dur_pred), min=1, max=duration_upper_limit).long()
        samples["locs"] = torch.cat(samples["locs"], dim=1)
        samples["durs"] = torch.cat(samples["durs"], dim=1)
        return samples
