from transformers import AutoConfig

# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

import torch
import torch.nn.functional as F

import torch.nn as nn

import math


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        hidden_t_dim,
        num_encoder_layers,
        hidden_size=768,
        num_attention_heads=12,
        dropout=0,
        max_location=None,
        learned_mean_embed=False,
    ):
        super().__init__()

        model_config = BertConfig()

        model_config.hidden_dropout_prob = dropout
        model_config.num_hidden_layers = num_encoder_layers
        model_config.hidden_size = hidden_size
        model_config.intermediate_size = hidden_size * 4
        model_config.max_position_embeddings = 768  # full dataset requires > 512
        model_config.num_attention_heads = num_attention_heads

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = input_dims
        self.dropout = dropout
        self.hidden_size = hidden_size

        # embeds
        self.token_embedding = nn.Embedding(max_location, self.input_dims)
        # self.dur_embedding = nn.Linear(1, self.input_dims)

        # network for encoding
        # self.embed_linear = nn.Linear(self.input_dims * 2, self.input_dims)
        # self.embed_final = nn.Linear(self.input_dims, self.input_dims)
        # self.embed_norm = nn.LayerNorm(self.input_dims)
        # self.embed_norm2 = nn.LayerNorm(self.input_dims)
        # self.linear1 = torch.nn.Linear(self.input_dims, self.input_dims * 4)
        # self.linear2 = torch.nn.Linear(self.input_dims * 4, self.input_dims)
        # self.dropout1 = nn.Dropout(p=0.1)
        # self.dropout2 = nn.Dropout(p=0.1)

        # network for decoding
        # self.out_norm = nn.LayerNorm(self.input_dims)
        # self.linear3 = torch.nn.Linear(self.input_dims, self.input_dims * 4)
        # self.linear4 = torch.nn.Linear(self.input_dims * 4, self.input_dims)
        # self.dropout3 = nn.Dropout(p=0.1)
        # self.dropout4 = nn.Dropout(p=0.1)

        # heads
        self.lm_head = nn.Linear(self.input_dims, max_location)
        # self.dur_head = nn.Linear(self.input_dims, 1)
        with torch.no_grad():
            self.lm_head.weight = self.token_embedding.weight
            # self.dur_head.weight = nn.Parameter(self.dur_embedding.weight.T)

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_t_dim, hidden_t_dim * 4),
            torch.nn.SiLU(),
            nn.Linear(hidden_t_dim * 4, model_config.hidden_size),
        )

        if self.input_dims != model_config.hidden_size:
            self.input_up_proj = nn.Sequential(
                nn.Linear(input_dims, model_config.hidden_size),
                nn.Tanh(),
                nn.Linear(model_config.hidden_size, model_config.hidden_size),
            )

        self.input_transformers = BertEncoder(model_config)

        self.register_buffer("position_ids", torch.arange(model_config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(model_config.max_position_embeddings, model_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)

        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)

        if self.output_dims != model_config.hidden_size:
            self.output_down_proj = nn.Sequential(
                nn.Linear(model_config.hidden_size, model_config.hidden_size),
                nn.Tanh(),
                nn.Linear(model_config.hidden_size, self.output_dims),
            )

        if learned_mean_embed:
            self.mean_embed = nn.Parameter(torch.randn(input_dims))
            nn.init.normal_(self.mean_embed, mean=0, std=input_dims**-0.5)
        else:
            self.mean_embed = None

    def _dense_block(self, x):
        x = self.linear2(self.dropout1(F.relu(self.linear1(x))))
        return self.dropout2(x)

    def _out_block(self, x):
        x = self.linear4(self.dropout3(F.relu(self.linear3(x))))
        return self.dropout4(x)

    def get_embeds(
        self,
        input_ids,
        # input_durs,
    ):
        # out = torch.cat([self.token_embedding(input_ids), self.dur_embedding(input_durs.unsqueeze(-1))], dim=-1)
        # out = self.embed_norm(self.embed_linear(out))

        # out = self.embed_norm2(out + self._dense_block(out))
        # return self.embed_final(out)
        return self.token_embedding(input_ids)

    def get_logits(self, hidden_repr):
        # out = self.out_norm(hidden_repr + self._out_block(hidden_repr))
        # return self.lm_head(out)
        return self.lm_head(hidden_repr)

    # def get_dur_predictions(self, hidden_repr):
    #     return self.dur_head(hidden_repr)

    def forward(self, x, timesteps, padding_mask):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        # the model
        input_trans_hidden_states = self.input_transformers(
            emb_inputs, attention_mask=padding_mask[:, None, None, :]
        ).last_hidden_state

        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
