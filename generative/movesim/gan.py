# coding: utf-8
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pickle


class Discriminator(nn.Module):
    """Basic discriminator."""

    def __init__(self, config, embedding, dropout=0.5):
        super(Discriminator, self).__init__()
        self.num_filters = [32, 32]
        self.kernel_sizes = [3, 5]

        self.embedding = embedding

        # changed it to account for the paddings of variable length
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n, (f, config.base_emb_size)) for (n, f) in zip(self.num_filters, self.kernel_sizes)]
        )
        self.highway = nn.Linear(sum(self.num_filters), sum(self.num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sum(self.num_filters), 1)

        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        padding_mask = x != 0  # batch_size * seq_len

        emb = self.embedding(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim

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
        embedding,
        device=None,
        starting_sample="rand",
        starting_dist=None,
    ):
        """

        :param starting_sample:
        :param starting_dist: prior knowledge of location preferences, could be empirical visit frequency. shape of [#locs]
        """
        super(Generator, self).__init__()
        self.base_emb_size = config.base_emb_size
        self.hidden_dim = config.hidden_dim
        self.total_loc_num = config.total_loc_num

        self.device = device
        self.starting_sample = starting_sample

        if self.starting_sample == "real":
            self.starting_dist = torch.tensor(starting_dist).float()

        # distance and empirical visits
        # self.dist_matrix = pickle.load(open(os.path.join(config.temp_save_root, "temp", "dist_matrix.pk"), "rb"))
        # self.emp_matrix = pickle.load(open(os.path.join(config.temp_save_root, "temp", "emp_matrix.pk"), "rb"))

        self.loc_embedding = embedding
        # self.tim_embedding = nn.Embedding(num_embeddings=24, embedding_dim=self.base_emb_size)

        # transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.base_emb_size, nhead=16, activation="gelu", dim_feedforward=self.hidden_dim, batch_first=True
        )
        encoder_norm = torch.nn.LayerNorm(self.base_emb_size)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
            norm=encoder_norm,
        )

        self.linear = nn.Linear(self.base_emb_size, self.total_loc_num)

        # for distance and empirical matrics
        # self.linear_mat1 = nn.Linear(self.total_loc_num - 1, self.base_emb_size)
        # self.linear_mat1_2 = nn.Linear(self.base_emb_size, self.total_loc_num)

        # self.linear_mat2 = nn.Linear(self.total_loc_num - 1, self.base_emb_size)
        # self.linear_mat2_2 = nn.Linear(self.base_emb_size, self.total_loc_num)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), True, dtype=torch.bool), diagonal=1)

    def forward(self, x_l):
        # x_t
        """

        :param x: (batch_size, seq_len), sequence of locations
        :return:
            (batch_size * seq_len, total_locations), prediction of next stage of all locations
        """
        src_padding_mask = (x_l == 0).to(self.device)

        # locs = x_l.reshape(-1).detach().cpu().numpy() - 1  # for padding
        # dist_vec = self.dist_matrix[locs]
        # visit_vec = self.emp_matrix[locs]
        # dist_vec = torch.Tensor(dist_vec).to(self.device)
        # visit_vec = torch.Tensor(visit_vec).to(self.device)

        emb = self.loc_embedding(x_l)

        #
        src_mask = self._generate_square_subsequent_mask(x_l.shape[1]).to(self.device)
        out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # for padding
        out = out * (~src_padding_mask).unsqueeze(-1).repeat(1, 1, self.base_emb_size)

        x = F.relu(self.linear(out.reshape(-1, self.base_emb_size)))

        # matrix calculation
        # dist_vec = F.relu(self.linear_mat1(dist_vec))
        # dist_vec = torch.sigmoid(self.linear_mat1_2(dist_vec))
        # dist_vec = F.normalize(dist_vec)

        # visit_vec = F.relu(self.linear_mat2(visit_vec))
        # visit_vec = torch.sigmoid(self.linear_mat2_2(visit_vec))
        # visit_vec = F.normalize(visit_vec)

        # pred = x + torch.mul(x, dist_vec) + torch.mul(x, visit_vec)

        pred = x
        return pred

    def step(self, input):
        """

        :param x: (batch_size, 1), current location
        :return:
            (batch_size, total_locations), prediction of next stage
        """
        src_padding_mask = (input == 0).to(self.device)

        # locs = input.reshape(-1).detach().cpu().numpy() - 1  # for padding
        # dist_vec = self.dist_matrix[locs]
        # visit_vec = self.emp_matrix[locs]
        # dist_vec = torch.Tensor(dist_vec).to(self.device)
        # visit_vec = torch.Tensor(visit_vec).to(self.device)

        emb = self.loc_embedding(input)

        #
        src_mask = self._generate_square_subsequent_mask(input.shape[1]).to(self.device)
        out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # for padding
        out = out * (~src_padding_mask).unsqueeze(-1).repeat(1, 1, self.base_emb_size)

        x = F.relu(self.linear(out.reshape(-1, self.base_emb_size)))

        # matrix calculation
        # dist_vec = F.relu(self.linear_mat1(dist_vec))
        # dist_vec = torch.sigmoid(self.linear_mat1_2(dist_vec))
        # dist_vec = F.normalize(dist_vec)

        # visit_vec = F.relu(self.linear_mat2(visit_vec))
        # visit_vec = torch.sigmoid(self.linear_mat2_2(visit_vec))
        # visit_vec = F.normalize(visit_vec)

        # pred = x + torch.mul(x, dist_vec) + torch.mul(x, visit_vec)
        pred = x

        return F.softmax(pred, dim=-1)

    def sample(self, batch_size, seq_len, x=None):
        """

        :param batch_size: int, size of a batch of training data
        :param seq_len: int, length of the sequence
        :param x: (batch_size, k), current generated sequence
        :return: (batch_size, seq_len), complete generated sequence
        """
        flag = False

        # self.attn.flatten_parameters()
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
                x[x == 0] += 1  # for padding
                s = 1

        samples = []
        if flag:
            if s > 0:
                samples.append(x)
            for i in range(s, seq_len):
                x = self.step(x)
                x = x.multinomial(1)
                x[x == 0] += 1  # for padding
                samples.append(x)
        else:
            given_len = x.size(1)

            lis = x.chunk(given_len, dim=1)
            for i in range(given_len):
                samples.append(lis[i])

            x = self.step(lis[-1])
            x = x.multinomial(1)
            x[x == 0] += 1  # for padding

            for i in range(given_len, seq_len):
                samples.append(x)
                x = self.step(x)
                x = x.multinomial(1)
                x[x == 0] += 1  # for padding
        return torch.cat(samples, dim=1)
