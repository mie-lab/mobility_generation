import torch.nn as nn
import torch
from torch import Tensor


from loc_predict.models.embed import AllEmbedding
from loc_predict.models.fc import FullyConnected


class TransEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransEncoder, self).__init__()

        self.d_input = config.base_emb_size
        self.Embedding = AllEmbedding(self.d_input, config)

        # encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.d_input, nhead=config.nhead, activation="gelu", dim_feedforward=config.dim_feedforward
        )
        encoder_norm = torch.nn.LayerNorm(self.d_input)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.num_encoder_layers, norm=encoder_norm, mask_check=True
        )

        self.fc = FullyConnected(self.d_input, config, if_residual_layer=True)

        self._init_weights()

    def forward(self, src, context_dict, device) -> Tensor:
        emb = self.Embedding(src, context_dict)
        seq_len = context_dict["len"]

        # positional encoding, dropout performed inside
        src_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        src_padding_mask = (src == 0).transpose(0, 1).to(device)
        out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # only take the last timestep
        out = out.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1,
        ).squeeze(0)

        return self.fc(out, context_dict["user"])

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf"), dtype=torch.bool), diagonal=1)

    def _init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
