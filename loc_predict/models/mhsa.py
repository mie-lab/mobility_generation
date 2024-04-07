import torch.nn as nn
import torch
from torch import Tensor


from loc_predict.models.embed import AllEmbedding
from loc_predict.models.fc import FullyConnected

from transformers import GPT2Config, GPT2Model


class TransEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransEncoder, self).__init__()

        self.d_input = config.base_emb_size
        self.Embedding = AllEmbedding(self.d_input, config, if_pos_encoder=False)

        # encoder
        configuration = GPT2Config()
        configuration.vocab_size = config.max_location
        configuration.n_embd = config.base_emb_size
        configuration.n_layer = config.num_encoder_layers
        configuration.n_head = config.nhead

        self.model = GPT2Model(configuration)
        self.model.set_input_embeddings(None)

        self.fc = FullyConnected(self.d_input, config, if_residual_layer=True)

        self._init_weights()

    def forward(self, src, context_dict) -> Tensor:
        emb = self.Embedding(src, context_dict)
        seq_len = context_dict["len"]

        # padding is 0
        src_padding_mask = (src != 0) * 1

        out = self.model(inputs_embeds=emb, attention_mask=src_padding_mask).last_hidden_state

        # only take the last timestep
        out = out.gather(1, seq_len.view([-1, 1, 1]).expand([out.shape[0], 1, out.shape[-1]]) - 1).squeeze()

        return self.fc(out, context_dict["user"])

    def _init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
