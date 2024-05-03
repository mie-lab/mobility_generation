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
        loaded_embed=None,
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
        if loaded_embed is not None:
            self.token_embedding = nn.Embedding.from_pretrained(torch.tensor(loaded_embed))
        else:
            self.token_embedding = nn.Embedding(max_location, self.input_dims)

        # heads
        self.lm_head = nn.Linear(self.input_dims, max_location)
        with torch.no_grad():
            self.lm_head.weight = self.token_embedding.weight

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_t_dim, hidden_t_dim * 4),
            torch.nn.SiLU(),
            nn.Linear(hidden_t_dim * 4, model_config.hidden_size),
        )

        # the residual
        self.input_up_proj_xy = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.attn = nn.MultiheadAttention(self.hidden_size, 8, batch_first=True)
        self.norm = nn.BatchNorm1d(self.hidden_size)
        self.linear1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout1 = nn.Dropout(p=0.1)

        self.input_up_proj = nn.Sequential(
            nn.Linear(input_dims, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.input_transformers = BertEncoder(model_config)

        self.register_buffer("position_ids", torch.arange(model_config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(model_config.max_position_embeddings, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=model_config.layer_norm_eps)

        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)

        if self.output_dims != self.hidden_size:
            self.output_down_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_dims),
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

    def get_embeds(self, input_ids):
        return self.token_embedding(input_ids)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def _res_block_combine(self, x, xy, key_padding_mask):
        emb_x = self.input_up_proj(x)
        emb_xy = self.input_up_proj_xy(xy)

        context, _ = self.attn(emb_x, emb_xy, emb_xy, key_padding_mask=key_padding_mask)

        emb_x = torch.cat([emb_x, context], 1)
        emb_x = self.dropout1(F.relu(self.linear1(emb_x)))

        # residual connection
        emb_x = self.norm(emb_x + self._ff_block(emb_x))
        return emb_x

        # x = self.linear2(self.dropout1(F.relu(self.linear1(x))))
        # return self.dropout2(x)

    def forward(self, x, xy, timesteps, padding_mask):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        emb_x = self._res_block_combine(x, xy, padding_mask)

        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
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
