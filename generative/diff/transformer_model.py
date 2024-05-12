from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

import torch
import torch.nn.functional as F

import torch.nn as nn

import math
import numpy as np


def _cal_freq_list(frequency_num, max_radius, min_radius):
    log_timescale_increment = math.log(float(max_radius) / float(min_radius)) / (frequency_num * 1.0 - 1)

    timescales = min_radius * np.exp(np.arange(frequency_num).astype(float) * log_timescale_increment)

    freq_list = 1.0 / timescales

    return freq_list


class TheoryGridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=400,
        min_radius=0.5,
        device="",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TheoryGridCellSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim
        self.max_radius = max_radius
        self.min_radius = min_radius

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = torch.tensor([1.0, 0.0]).to(device)  # 0
        self.unit_vec2 = torch.tensor([-1.0 / 2.0, math.sqrt(3) / 2.0]).to(device)  # 120 degree
        self.unit_vec3 = torch.tensor([-1.0 / 2.0, -math.sqrt(3) / 2.0]).to(device)  # 240 degree

        # the frequence we use for each block, alpha in ICLR paper
        freq_list = _cal_freq_list(self.frequency_num, self.max_radius, self.min_radius)
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = torch.tensor(np.repeat(freq_mat, 6, axis=1)).to(device).float()

    def make_input_embeds(self, coords_mat):
        # (batch_size, num_context_pt, coord_dim)
        batch_size, num_pt, _ = coords_mat.shape

        # compute the dot product between [deltaX, deltaY] and each unit_vec
        # (batch_size, num_context_pt, 1)
        angle_mat1 = torch.unsqueeze(torch.matmul(coords_mat, self.unit_vec1), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = torch.unsqueeze(torch.matmul(coords_mat, self.unit_vec2), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = torch.unsqueeze(torch.matmul(coords_mat, self.unit_vec3), axis=-1)

        # (batch_size, num_context_pt, 6)
        angle_mat = torch.cat([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis=-1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = torch.unsqueeze(angle_mat, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = torch.repeat_interleave(angle_mat, self.frequency_num, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = torch.reshape(angle_mat, (batch_size, num_pt, -1))

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = torch.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = torch.cos(spr_embeds[:, :, 1::2])  # dim 2i+1

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        return spr_embeds


class ContextModel(nn.Module):
    def __init__(self, input_dims, hidden_dims, embed_xy, embed_poi, poi_dims=None, device=""):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.embed_xy = embed_xy
        self.embed_poi = embed_poi

        # upproject embedding
        self.input_up_proj = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
        )
        # xy embedding
        if embed_xy:
            frequency_num = int(hidden_dims / 6)
            self.encoder = TheoryGridCellSpatialRelationEncoder(frequency_num=frequency_num, device=device)

        # poi embedding
        if embed_poi:
            self.poi_up_proj = nn.Sequential(
                nn.Linear(poi_dims, input_dims),
                nn.Tanh(),
                nn.Linear(input_dims, input_dims),
            )

    def forward(self, x, context):
        emb = self.input_up_proj(x)
        if self.embed_xy:
            emb = emb + self.encoder(context["xy"])
        if self.embed_poi:
            emb = emb + self.poi_up_proj(context["poi"])
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
        device="",
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
            nn.SiLU(),
            nn.Linear(input_dims * 4, self.hidden_size),
        )

        self.context_model = ContextModel(
            self.input_dims, self.hidden_size, embed_xy, embed_poi, poi_dims=poi_dims, device=device
        )

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
        emb_x = self.context_model(x, context)

        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        #
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        # the model
        input_trans_hidden_states = self.input_transformers(
            emb_inputs, attention_mask=padding_mask[:, None, None, :]
        ).last_hidden_state

        h = self.output_down_proj(input_trans_hidden_states)
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
