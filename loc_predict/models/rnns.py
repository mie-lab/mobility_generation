import torch
import torch.nn as nn


import torch.nn.functional as F
from torch import Tensor

from loc_predict.models.embed import AllEmbedding
from loc_predict.models.fc import FullyConnected


class RNNs(nn.Module):
    """Baseline LSTM model."""

    def __init__(self, config):
        super(RNNs, self).__init__()
        self.attention = config.attention
        self.d_input = config.base_emb_size
        self.out_dim = config.hidden_size

        self.Embedding = AllEmbedding(self.d_input, config, if_pos_encoder=False)

        self.model = RNN_Classifier(self.d_input, config)

        if self.attention:
            self.attn = DotProductAttention()

            self.norm = nn.BatchNorm1d(self.out_dim)
            self.linear1 = torch.nn.Linear(self.out_dim * 2, self.out_dim)

            self.linear2 = torch.nn.Linear(self.out_dim, self.out_dim * 2)
            self.linear3 = torch.nn.Linear(self.out_dim * 2, self.out_dim)
            self.dropout1 = nn.Dropout(p=0.1)
            self.dropout2 = nn.Dropout(p=0.1)
            self.dropout3 = nn.Dropout(p=0.1)

        self.fc = FullyConnected(self.out_dim, config, if_residual_layer=False)

        self._init_weights_rnn()

    def forward(self, src, context_dict, device):
        emb = self.Embedding(src, context_dict)
        seq_len = context_dict["len"]

        # model
        out, _ = self.model(emb)

        # only take the last timestep
        last = out.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1,
        ).squeeze(0)

        if self.attention:
            context, _ = self.attn(last, out, out, seq_len, device)

            last = torch.cat([last, context], 1)
            last = self.dropout1(F.relu(self.linear1(last)))

            # residual connection
            last = self.norm(last + self._ff_block(last))

        return self.fc(last)

    def _init_weights_rnn(self):
        """Reproduce Keras default initialization weights for consistency with Keras version."""

        ih = (param.data for name, param in self.named_parameters() if "weight_ih" in name)
        hh = (param.data for name, param in self.named_parameters() if "weight_hh" in name)
        b = (param.data for name, param in self.named_parameters() if "bias" in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear3(self.dropout2(F.relu(self.linear2(x))))
        return self.dropout3(x)


class RNN_Classifier(nn.Module):
    """Baseline LSTM model."""

    def __init__(self, d_input, config):
        super(RNN_Classifier, self).__init__()

        RNNS = ["LSTM", "GRU"]
        self.bidirectional = False
        assert config.rnn_type in RNNS, "Use one of the following: {}".format(str(RNNS))
        rnn_cell = getattr(nn, config.rnn_type)  # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(
            d_input, hidden_size=config.hidden_size, num_layers=1, dropout=0.0, bidirectional=self.bidirectional
        )

    def forward(self, input, hidden=None):
        """Forward pass of the network."""
        return self.rnn(input, hidden)


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query, keys, values, src_len, device):
        """
        Here we assume q_dim == k_dim (dot product attention).

        Query = [BxQ]
        Keys = [TxBxK]
        Values = [TxBxV]
        Outputs = a:[TxB], lin_comb:[BxV]
        src_len:
           used for masking. NoneType or tensor in shape (B) indicating sequence length
        """
        keys = keys.transpose(0, 1)  # [B*T*H]
        attn = torch.bmm(keys, query.unsqueeze(-1)).transpose(2, 1)  # [B,1,T]

        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (keys.size(1) - src_len[b].item()))
            mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(1).to(device)  # [B,1,T]
            attn.masked_fill_(mask, -1e18)
        attn = F.softmax(attn, dim=2)  # scale, normalize

        values = values.transpose(0, 1)  # [TxBxV] -> [BxTxV]
        context = torch.bmm(attn, values).squeeze(1)  # [Bx1xT]x[BxTxV] -> [BxV]
        return context, attn
