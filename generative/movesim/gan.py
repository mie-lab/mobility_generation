# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """Basic discriminator."""

    def __init__(self, config, embedding, dropout=0.5):
        super(Discriminator, self).__init__()
        num_filters = [64, 64, 64, 64, 64, 64, 64, 64]
        filter_sizes = [1, 3, 3, 3, 3, 3, 3, 3]

        self.embedding = embedding

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n, (f, config.base_emb_size)) for (n, f) in zip(num_filters, filter_sizes)]
        )
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sum(num_filters), 1)

        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.embedding(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        # [batch_size * num_filter * length]
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
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

        self.loc_embedding = embedding
        # self.tim_embedding = nn.Embedding(num_embeddings=24, embedding_dim=self.base_emb_size)

        self.attn = nn.MultiheadAttention(self.hidden_dim, 4)
        self.Q = nn.Linear(self.base_emb_size, self.hidden_dim)
        self.V = nn.Linear(self.base_emb_size, self.hidden_dim)
        self.K = nn.Linear(self.base_emb_size, self.hidden_dim)

        self.attn2 = nn.MultiheadAttention(self.hidden_dim, 1)
        self.Q2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear = nn.Linear(self.hidden_dim, config.total_loc_num)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def init_hidden(self, batch_size):
        h = torch.LongTensor(torch.zeros((1, batch_size, self.hidden_dim)))
        c = torch.LongTensor(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.device:
            h, c = h.to(self.device), c.to(self.device)
        return h, c

    def forward(self, x_l):
        # x_t
        """

        :param x: (batch_size, seq_len), sequence of locations
        :return:
            (batch_size * seq_len, total_locations), prediction of next stage of all locations
        """
        lemb = self.loc_embedding(x_l)
        # temb = self.tim_embedding(x_t)
        # x = lemb + temb
        x = lemb

        x = x.transpose(0, 1)
        Query = self.Q(x)
        Query = F.relu(Query)

        Value = self.V(x)
        Value = F.relu(Value)

        Key = self.K(x)
        Key = F.relu(Key)

        x, _ = self.attn(Query, Key, Value)

        Query = self.Q2(x)
        Query = F.relu(Query)

        Value = self.V2(x)
        Value = F.relu(Value)

        Key = self.K2(x)
        Key = F.relu(Key)

        x, _ = self.attn2(Query, Key, Value)

        x = x.transpose(0, 1)

        x = x.reshape(-1, self.hidden_dim)
        x = self.linear(x)

        return F.relu(x)

    def step(self, input):
        """

        :param x: (batch_size, 1), current location
        :param h: (1/2, batch_size, hidden_dim), lstm hidden state
        :param c: (1/2, batch_size, hidden_dim), lstm cell state
        :return:
            (batch_size, total_locations), prediction of next stage
        """

        lemb = self.loc_embedding(input)
        # temb = self.tim_embedding(t)
        # x = lemb + temb
        x = lemb

        x = x.transpose(0, 1)

        Query = self.Q(x)
        Query = F.relu(Query)

        Value = self.V(x)
        Value = F.relu(Value)

        Key = self.K(x)
        Key = F.relu(Key)

        x, _ = self.attn(Query, Key, Value)

        Query = self.Q2(x)
        Query = F.relu(Query)

        Value = self.V2(x)
        Value = F.relu(Value)

        Key = self.K2(x)
        Key = F.relu(Key)

        x, _ = self.attn2(Query, Key, Value)

        x = x.transpose(0, 1)

        x = x.reshape(-1, self.hidden_dim)
        x = self.linear(x)
        x = F.relu(x)

        return F.softmax(x, dim=-1)

    def sample(self, batch_size, seq_len, x=None):
        """

        :param batch_size: int, size of a batch of training data
        :param seq_len: int, length of the sequence
        :param x: (batch_size, k), current generated sequence
        :return: (batch_size, seq_len), complete generated sequence
        """
        # self.attn.flatten_parameters()

        s = 0
        if x is None:
            if self.starting_sample == "rand":
                x = torch.LongTensor(torch.randint(high=self.total_loc_num, size=(batch_size, 1))).to(self.device)
            elif self.starting_sample == "real":
                x = torch.LongTensor(
                    torch.stack([torch.multinomial(self.starting_dist, 1) for _ in range(batch_size)], dim=0)
                ).to(self.device)
                s = 1

        samples = []
        if x is None:
            if s > 0:
                samples.append(x)
            for i in range(s, seq_len):
                # t = torch.LongTensor([i % 24]).to(self.device)
                # t = t.repeat(batch_size).reshape(batch_size, -1)
                x = self.step(x)
                x = x.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                # t = torch.LongTensor([i % 24]).to(self.device)
                # t = t.repeat(batch_size).reshape(batch_size, -1)
                x = self.step(lis[i])
                samples.append(lis[i])
            x = x.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                # t = torch.LongTensor([i % 24]).to(self.device)
                # t = t.repeat(batch_size).reshape(batch_size, -1)
                x = self.step(x)
                x = x.multinomial(1)
        return torch.cat(samples, dim=1)
