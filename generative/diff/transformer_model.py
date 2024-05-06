from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

import torch
import torch.nn.functional as F

import torch.nn as nn

import math


class ContextModel(nn.Module):
    def __init__(self, input_dims, embed_xy, embed_poi, poi_dims=None):
        super().__init__()

        self.input_dims = input_dims
        self.embed_xy = embed_xy
        self.embed_poi = embed_poi

        if embed_xy:
            # xy embedding
            self.xy_up_proj = nn.Sequential(
                nn.Linear(2, input_dims),
                nn.Tanh(),
                nn.Linear(input_dims, input_dims),
            )
            self.mha_xy = nn.MultiheadAttention(input_dims, 8, batch_first=True)
            self.comb_xy = nn.Sequential(
                nn.Linear(input_dims * 2, input_dims),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
        if embed_poi:
            # poi embedding
            self.poi_up_proj = nn.Sequential(
                nn.Linear(poi_dims, input_dims),
                nn.Tanh(),
                nn.Linear(input_dims, input_dims),
            )
            self.mha_poi = nn.MultiheadAttention(input_dims, 8, batch_first=True)

            self.comb_poi = nn.Sequential(
                nn.Linear(input_dims * 2, input_dims),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
        if embed_xy and embed_poi:
            self.comb_xy_poi = nn.Sequential(
                nn.Linear(input_dims * 2, input_dims),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
        if embed_xy or embed_poi:
            self.norm = nn.LayerNorm(input_dims)
            self.ff_block = nn.Sequential(
                nn.Linear(input_dims, input_dims * 4),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(input_dims * 4, input_dims),
                nn.Dropout(p=0.1),
            )

    def forward(self, x, context, key_padding_mask):
        emb = x
        if self.embed_xy:
            xy = self.xy_up_proj(context["xy"])
            attn_xy, _ = self.mha_xy(emb, xy, xy, key_padding_mask=(1 - key_padding_mask).to(bool))
            emb_xy = self.comb_xy(torch.cat([emb, attn_xy], -1))

        if self.embed_poi:
            poi = self.poi_up_proj(context["poi"])
            attn_poi, _ = self.mha_poi(emb, poi, poi, key_padding_mask=(1 - key_padding_mask).to(bool))
            emb_poi = self.comb_poi(torch.cat([emb, attn_poi], -1))

        if self.embed_xy and self.embed_poi:
            emb = self.comb_xy_poi(torch.cat([emb_xy, emb_poi], -1))
        elif self.embed_xy:
            emb = emb_xy
        elif self.embed_poi:
            emb = emb_poi

        # residual connection
        if self.embed_xy or self.embed_poi:
            emb = self.norm(emb + self.ff_block(emb))
        return emb


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
        num_encoder_layers,
        hidden_size=768,
        num_attention_heads=12,
        dropout=0,
        max_location=None,
        learned_mean_embed=False,
        loaded_embed=None,
        embed_xy=False,
        embed_poi=False,
        poi_dims=32,
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
        self.output_dims = input_dims
        self.dropout = dropout
        self.hidden_size = hidden_size

        # embeds and heads
        if loaded_embed is not None:
            self.token_embedding = nn.Embedding.from_pretrained(torch.tensor(loaded_embed))
        else:
            self.token_embedding = nn.Embedding(max_location, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, max_location)
        with torch.no_grad():
            self.lm_head.weight = self.token_embedding.weight

        # timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(input_dims, input_dims * 4),
            torch.nn.SiLU(),
            nn.Linear(input_dims * 4, self.hidden_size),
        )
        # upproject embedding
        self.input_up_proj = nn.Sequential(
            nn.Linear(input_dims, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.context_model = ContextModel(self.input_dims, embed_xy, embed_poi, poi_dims)

        self.input_transformers = BertEncoder(model_config)

        self.register_buffer("position_ids", torch.arange(model_config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(model_config.max_position_embeddings, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=model_config.layer_norm_eps)

        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)

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

    def get_embeds(self, input_ids):
        return self.token_embedding(input_ids)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, x, context, timesteps, padding_mask):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.input_dims))

        # combine x and context (xy and poi)
        emb_x = self.context_model(x, context, padding_mask)
        emb_x = self.input_up_proj(emb_x)

        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        # the model
        input_trans_hidden_states = self.input_transformers(
            emb_inputs, attention_mask=padding_mask[:, None, None, :]
        ).last_hidden_state

        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)

        return h

        # pred_xy = self.output_down_proj_xy(input_trans_hidden_states)
        # pred_xy = pred_xy.type(x.dtype)
        # return h, pred_xy


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
