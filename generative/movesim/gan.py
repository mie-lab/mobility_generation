# coding: utf-8

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
        # self.highway = nn.Linear(sum(self.num_filters), sum(self.num_filters))
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
        # highway = self.highway(pred)
        # pred = F.sigmoid(highway) * F.relu(highway) + (1.0 - F.sigmoid(highway)) * pred

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
        # self.dist_matrix = pickle.load(open("./data/temp/dist_matrix.pk", "rb"))
        # self.emp_matrix = pickle.load(open("./data/temp/emp_matrix.pk", "rb"))

        self.loc_embedding = embedding
        # self.tim_embedding = nn.Embedding(num_embeddings=24, embedding_dim=self.base_emb_size)

        self.attn = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True)
        self.Q = nn.Linear(self.base_emb_size, self.hidden_dim)
        self.V = nn.Linear(self.base_emb_size, self.hidden_dim)
        self.K = nn.Linear(self.base_emb_size, self.hidden_dim)

        self.attn2 = nn.MultiheadAttention(self.hidden_dim, 4, batch_first=True)
        self.Q2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear = nn.Linear(self.hidden_dim, self.total_loc_num)

        # for distance and empirical matrics
        # self.linear_mat1 = nn.Linear(self.total_loc_num - 1, self.hidden_dim)
        # self.linear_mat1_2 = nn.Linear(self.hidden_dim, self.total_loc_num)

        # self.linear_mat2 = nn.Linear(self.total_loc_num - 1, self.hidden_dim)
        # self.linear_mat2_2 = nn.Linear(self.hidden_dim, self.total_loc_num)

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
        src_padding_mask = x_l == 0

        # locs = x_l.reshape(-1).detach().cpu().numpy() - 1  # for padding
        # dist_vec = self.dist_matrix[locs]
        # visit_vec = self.emp_matrix[locs]
        # dist_vec = torch.Tensor(dist_vec).to(self.device)
        # visit_vec = torch.Tensor(visit_vec).to(self.device)

        x = self.loc_embedding(x_l)

        Query = F.relu(self.Q(x))
        Value = F.relu(self.V(x))
        Key = F.relu(self.K(x))

        x, _ = self.attn(Query, Key, Value, key_padding_mask=src_padding_mask)

        Query = F.relu(self.Q2(x))
        Value = F.relu(self.V2(x))
        Key = F.relu(self.K2(x))

        x, _ = self.attn2(Query, Key, Value, key_padding_mask=src_padding_mask)

        x = F.relu(self.linear(x.reshape(-1, self.hidden_dim)))

        # matrix calculation
        # dist_vec = F.relu(self.linear_mat1(dist_vec))
        # dist_vec = F.sigmoid(self.linear_mat1_2(dist_vec))
        # dist_vec = F.normalize(dist_vec)

        # visit_vec = F.relu(self.linear_mat2(visit_vec))
        # visit_vec = F.sigmoid(self.linear_mat2_2(visit_vec))
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
        src_padding_mask = input == 0

        # locs = input.reshape(-1).detach().cpu().numpy() - 1  # for padding
        # dist_vec = self.dist_matrix[locs]
        # visit_vec = self.emp_matrix[locs]
        # dist_vec = torch.Tensor(dist_vec).to(self.device)
        # visit_vec = torch.Tensor(visit_vec).to(self.device)

        x = self.loc_embedding(input)

        Query = F.relu(self.Q(x))
        Value = F.relu(self.V(x))
        Key = F.relu(self.K(x))

        attn, _ = self.attn(Query, Key, Value, key_padding_mask=src_padding_mask)

        Query = F.relu(self.Q2(attn))
        Value = F.relu(self.V2(attn))
        Key = F.relu(self.K2(attn))

        attn, _ = self.attn2(Query, Key, Value, key_padding_mask=src_padding_mask)

        x = F.relu(self.linear(attn.reshape(-1, self.hidden_dim)))

        # matrix calculation
        # dist_vec = F.relu(self.linear_mat1(dist_vec))
        # dist_vec = F.sigmoid(self.linear_mat1_2(dist_vec))
        # dist_vec = F.normalize(dist_vec)

        # visit_vec = F.relu(self.linear_mat2(visit_vec))
        # visit_vec = F.sigmoid(self.linear_mat2_2(visit_vec))
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
                x[x == 0] += 1
                s = 1

        samples = []
        if flag:
            if s > 0:
                samples.append(x)
            for i in range(s, seq_len):
                x = self.step(x)
                x = x.multinomial(1)
                x[x == 0] += 1
                samples.append(x)
        else:
            given_len = x.size(1)

            lis = x.chunk(given_len, dim=1)
            for i in range(given_len):
                samples.append(lis[i])

            x = self.step(lis[-1])
            x = x.multinomial(1)
            x[x == 0] += 1

            for i in range(given_len, seq_len):
                samples.append(x)
                x = self.step(x)
                x = x.multinomial(1)
                x[x == 0] += 1
        return torch.cat(samples, dim=1)
