from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

import torch
import torch.nn.functional as F

import torch.nn as nn

import math
import numpy as np

from improved_diffusion.gaussian_diffusion import GaussianDiffusion, betas_for_alpha_bar
from improved_diffusion.respace import SpacedDiffusion, space_timesteps


def _cal_freq_list(frequency_num, max_radius, min_radius):
    log_timescale_increment = math.log(float(max_radius) / float(min_radius)) / (frequency_num * 1.0 - 1)

    timescales = min_radius * np.exp(np.arange(frequency_num).astype(float) * log_timescale_increment)

    freq_list = 1.0 / timescales

    return freq_list


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


class TheoryGridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(
        self,
        coord_dim=2,
        frequency_num=16,
        max_radius=350,
        min_radius=1,
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
        self.input_up_proj = nn.Linear(input_dims, hidden_dims, bias=False)
        # xy embedding
        if embed_xy:
            frequency_num = 16
            self.encoder = TheoryGridCellSpatialRelationEncoder(frequency_num=frequency_num, device=device)
            self.comb_xy = nn.Sequential(
                nn.Linear(hidden_dims + frequency_num * 6, input_dims),
                nn.LayerNorm(input_dims),
                nn.ReLU(),
                nn.Linear(input_dims, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.Dropout(0.1),
            )

        # poi embedding
        if embed_poi:
            self.poi_up_proj = nn.Sequential(
                nn.Linear(poi_dims, input_dims),
                nn.Tanh(),
                nn.Linear(input_dims, input_dims),
            )
            self.comb_poi = nn.Sequential(
                nn.Linear(hidden_dims + input_dims, input_dims),
                nn.LayerNorm(input_dims),
                nn.ReLU(),
                nn.Linear(input_dims, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.Dropout(0.1),
            )

    def forward(self, x, context):
        emb = self.input_up_proj(x)
        if self.embed_xy:
            res = torch.cat([emb, self.encoder(context["xy"])], dim=-1)
            emb = emb + self.comb_xy(res)
        if self.embed_poi:
            res = torch.cat([emb, self.poi_up_proj(context["poi"])], dim=-1)
            emb = emb + self.comb_poi(res)
        return emb


class TransEncoder(nn.Module):
    def __init__(
        self,
        num_encoder_layers=2,
        hidden_size=768,
        num_attention_heads=12,
        dropout=0,
        location_embedding=None,
        position_embedding=None,
        input_up_proj=None,
        # device="",
    ):
        super().__init__()

        self.padding_idx = location_embedding.padding_idx
        self.location_embedding = location_embedding

        self.embed_scale = math.sqrt(location_embedding.embedding_dim)

        # up projection (shared)
        self.input_up_proj = input_up_proj

        # position embeddings (shared)
        max_source_positions = 512
        self.register_buffer("position_ids", torch.arange(max_source_positions).expand((1, -1)))
        self.position_embedding = position_embedding

        # # context model for embedding
        # self.context_model = ContextModel(
        #     self.input_dims, self.hidden_size, embed_xy, embed_poi, poi_dims=poi_dims, device=device
        # )

        #
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_attention_heads)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

    def forward(self, src_tokens, context=None):
        x = self.forward_embedding(src_tokens, context)

        # B x T, True will be ignored
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder model
        hidden = self.model(x, src_key_padding_mask=encoder_padding_mask)

        return {
            "encoder_out": hidden,  # T x B x C
            "encoder_padding_mask": encoder_padding_mask,  # B x T
        }

    def forward_embedding(self, src_tokens, context=None):
        token_embedding = self.location_embedding(src_tokens)
        x = self.embed_scale * token_embedding

        # up-projection
        x = self.input_up_proj(x)

        # position_embeddings
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        x = x + self.position_embedding(position_ids)

        #
        x = self.dropout(self.LayerNorm(x))

        return x


class TransDecoder(nn.Module):
    def __init__(
        self,
        num_decoder_layers=2,
        hidden_size=768,
        num_attention_heads=12,
        dropout=0,
        location_embedding=None,
        position_embedding=None,
        input_up_proj=None,
        output_down_proj=None,
    ):
        super().__init__()

        # up projection (shared)
        self.input_up_proj = input_up_proj
        self.output_down_proj = output_down_proj

        self.location_embedding = location_embedding
        self.embed_scale = math.sqrt(location_embedding.embedding_dim)

        self.hidden_size = hidden_size
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        # position embeddings (shared)
        max_source_positions = 512
        self.register_buffer("position_ids", torch.arange(max_source_positions).expand((1, -1)))
        self.position_embedding = position_embedding

        #
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        #
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_attention_heads)
        decoder_norm = nn.LayerNorm(hidden_size)
        self.model = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)

    def forward_embedding(self, tgt_tokens):
        # up-projection
        embed = self.embed_scale * self.location_embedding(tgt_tokens)

        return embed

    def forward_hidden(self, z_t, t):
        seq_length = z_t.size(1)
        #
        hidden = self.input_up_proj(z_t)

        # time embedding
        timestep_embed = timestep_embedding(t, self.hidden_size)
        emb_t = self.time_embed(timestep_embed)
        hidden = hidden + emb_t.unsqueeze(1).expand(-1, seq_length, -1)

        # position embedding
        position_ids = self.position_ids[:, :seq_length]
        hidden = hidden + self.position_embedding(position_ids)

        # embedding normalization
        hidden = self.dropout(self.LayerNorm(hidden))
        return hidden

    def forward(self, z_t, t, padding_mask, encoder_out):
        hidden = self.forward_hidden(z_t, t)

        # B x T x C -> T x B x C
        hidden = hidden.transpose(0, 1)

        hidden = self.model(
            tgt=hidden,
            memory=encoder_out["encoder_out"],
            tgt_key_padding_mask=~padding_mask,
            memory_key_padding_mask=encoder_out["encoder_padding_mask"],
        )

        # T x B x C -> B x T x C
        hidden = hidden.transpose(0, 1)

        hidden = self.output_down_proj(hidden)
        return hidden


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
        model_args=None,
        diff_args=None,
    ):
        super().__init__()

        self.model_args = model_args
        self.diff_args = diff_args

        max_positions = 512

        # location embedding
        self.location_embedding = nn.Embedding(model_args.max_location, model_args.input_dims, padding_idx=0)
        self.input_up_proj = nn.Linear(model_args.input_dims, model_args.hidden_size, bias=False)
        # position embedding
        self.position_embedding = nn.Embedding(max_positions, model_args.hidden_size)

        # out
        self.output_down_proj = nn.Linear(model_args.hidden_size, model_args.input_dims, bias=False)

        # encoder
        self.encoder = TransEncoder(
            num_encoder_layers=model_args.num_layers,
            hidden_size=model_args.hidden_size,
            num_attention_heads=model_args.num_attention_heads,
            dropout=model_args.dropout,
            location_embedding=self.location_embedding,
            position_embedding=self.position_embedding,
            input_up_proj=self.input_up_proj,
        )

        # decoder
        self.decoder = TransDecoder(
            num_decoder_layers=model_args.num_layers,
            hidden_size=model_args.hidden_size,
            num_attention_heads=model_args.num_attention_heads,
            dropout=model_args.dropout,
            location_embedding=self.location_embedding,
            position_embedding=self.position_embedding,
            input_up_proj=self.input_up_proj,
            output_down_proj=self.output_down_proj,
        )

        self.lm_head = nn.Linear(model_args.input_dims, model_args.max_location, bias=False)
        with torch.no_grad():
            self.lm_head.weight = self.location_embedding.weight

        self.training_diffusion = GaussianDiffusion(
            betas=get_named_beta_schedule(
                diff_args.noise_schedule,
                diff_args.diffusion_steps,
                diff_args.rescaling_factor if diff_args.vp_rf else 1.0,
            ),
            model_mean_type=None,
            model_var_type=None,
            loss_type=None,
        )

        # so we have different schedules in training and decoding
        self.decoding_diffusion = SpacedDiffusion(
            space_timesteps(diff_args.diffusion_steps, str(diff_args.decoding_steps)),
            betas=get_named_beta_schedule(
                diff_args.decoding_noise_schedule,
                diff_args.diffusion_steps,
                diff_args.decoding_rescaling_factor if diff_args.decoding_vp_rf else 1.0,
            ),
            model_mean_type=None,
            model_var_type=None,
            loss_type=None,
        )

        self.timesteps_scale = (1000.0 / diff_args.diffusion_steps) if diff_args.rescale_timesteps else 1.0

    def get_embeds(self, input_ids):
        return self.location_embedding(input_ids)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def forward(self, src, tgt, src_ctx, tgt_cxt, t):
        """Compute training losses"""

        # encoding
        encoder_out = self.encoder(src)

        # padding_mask B x T
        mask = tgt.ne(0)

        # diffusion
        z_0 = self.decoder.forward_embedding(tgt)
        model_t = t * self.timesteps_scale

        noise = torch.randn_like(z_0) * self.diff_args.rescaling_factor
        # reparametrization trick
        z_t = self.training_diffusion.q_sample(z_0, t, noise).type_as(z_0)

        # model use z_t to predict z_0
        z_0_hat = self.decoder(z_t, model_t, mask, encoder_out)
        logits = self.get_logits(z_0 if self.diff_args.rounding_loss else z_0_hat)

        terms = {}

        # Lt
        terms["mse"] = mean_flat((z_0_hat - z_0).square(), mask)

        # Rounding error: embedding regularization
        decoder_nll = token_discrete_loss(logits, tgt, mask=mask, label_smoothing=0)

        terms["loss"] = terms["mse"] + decoder_nll

        return terms

    def forward_decoder(self, z_t, step, mask, encoder_out, prev_z_0_hat=None):
        """Sample z_{t-1} given z_t"""

        # rescale timesteps
        model_t = self.decoding_diffusion.timestep_map[step] * self.timesteps_scale
        model_t = torch.full([len(z_t)], model_t, device=z_t.device)

        # predict z_0
        z_0_hat = self.decoder(z_t, model_t, mask, encoder_out)

        # clamping trick
        if self.diff_args.clamping:
            tokens = self.get_logits(z_0_hat).argmax(-1)
            z_0_hat = self.decoder.forward_embedding(tokens)

        # sample z_{t-1}
        t = torch.tensor(step, device=z_t.device)
        mean, _, log_variance = self.decoding_diffusion.q_posterior_mean_variance(z_0_hat, z_t, t)
        noise = torch.randn_like(z_t) * self.diff_args.decoding_rescaling_factor

        z_t = mean + (0.5 * log_variance).exp() * noise
        z_t = z_t.type_as(z_0_hat)

        return z_t, z_0_hat

    def forward_output_layer(self, z_t):
        scores, tokens = self.get_logits(z_t).log_softmax(-1).max(-1)
        return tokens, scores


def token_discrete_loss(logits, input_ids, mask=None, label_smoothing=0):
    """
    the loss of -log p(w|z_0)
    :param x_start_mean: word embedding
    :return: x_0
    """
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=0, label_smoothing=label_smoothing)
    decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
    if mask is not None:
        decoder_nll *= mask
        decoder_nll = decoder_nll.sum(dim=-1) / mask.sum(dim=-1)
    else:
        decoder_nll = decoder_nll.mean(dim=-1)

    return decoder_nll


def mean_flat(tensor, padding_mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    if padding_mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        padding_mask = torch.broadcast_to(padding_mask.unsqueeze(dim=-1), tensor.shape)
        tensor *= padding_mask
        return tensor.sum(dim=list(range(1, len(tensor.shape)))) / padding_mask.sum(
            dim=list(range(1, len(tensor.shape)))
        )


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, rescaling_factor=1):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """

    if schedule_name == "sqrt":
        shift = 0.0001
        alpha_bar = lambda t: 1 - np.sqrt(t + shift)

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    f2 = rescaling_factor**2
    rescaled_alpha_bar = lambda t: alpha_bar(t) / (f2 - (f2 - 1) * alpha_bar(t))

    return betas_for_alpha_bar(num_diffusion_timesteps, rescaled_alpha_bar)
