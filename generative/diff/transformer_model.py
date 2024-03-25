from transformers import AutoConfig

# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import GPT2Config, GPT2Model

import torch

import torch.nn as nn

import math


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/model_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        hidden_t_dim,
        dropout=0,
        model_name="bert-base-uncased",
        max_location=None,
    ):
        super().__init__()

        model_config = AutoConfig.from_pretrained(model_name)

        model_config.hidden_dropout_prob = dropout
        model_config.num_hidden_layers = 2
        model_config.hidden_size = 64
        model_config.intermediate_size = 256
        model_config.num_attention_heads = 4

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = input_dims
        self.dropout = dropout
        self.hidden_size = model_config.hidden_size

        self.word_embedding = nn.Embedding(max_location, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, max_location)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight

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

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, x, timesteps):
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

        # TODO: Padding masks?
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state

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
