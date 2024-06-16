from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

import torch
import torch.nn.functional as F

import torch.nn as nn

import math
import numpy as np
from random import random
import inspect

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
    def __init__(self, input_dims, hidden_dims, embed_xy, embed_poi, poi_dim=None, device=""):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.embed_xy = embed_xy
        self.embed_poi = embed_poi

        # xy embedding
        if embed_xy:
            frequency_num = 16
            self.encoder = TheoryGridCellSpatialRelationEncoder(frequency_num=frequency_num, device=device)
            self.comb_xy = nn.Sequential(
                nn.Linear(hidden_dims + frequency_num * 6, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.Dropout(0.1),
            )

        # poi embedding
        if embed_poi:
            self.poi_up_proj = nn.Linear(poi_dim, input_dims)
            self.comb_poi = nn.Sequential(
                nn.Linear(hidden_dims + input_dims, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.Dropout(0.1),
            )

    def forward(self, x, context):
        if self.embed_xy:
            res = torch.cat([x, self.encoder(context["xy"])], dim=-1)
            x = x + self.comb_xy(res)
        if self.embed_poi:
            res = torch.cat([x, self.poi_up_proj(context["poi"])], dim=-1)
            x = x + self.comb_poi(res)
        return x


class TransEncoder(nn.Module):
    def __init__(
        self,
        num_encoder_layers=2,
        hidden_size=768,
        num_attention_heads=12,
        dropout=0,
        location_embedding=None,
        mode_embedding=None,
        duration_embedding=None,
        position_embedding=None,
        input_up_proj=None,
        embed_xy=False,
        embed_poi=False,
        poi_dim=32,
        device="",
    ):
        super().__init__()

        self.padding_idx = location_embedding.padding_idx

        self.location_embedding = location_embedding
        self.embed_scale = math.sqrt(location_embedding.embedding_dim)
        #
        self.duration_embedding = duration_embedding
        #
        self.mode_embedding = mode_embedding

        # up projection (shared)
        self.input_up_proj = input_up_proj

        # position embeddings (shared)
        max_positions = 512
        self.register_buffer("position_ids", torch.arange(max_positions).expand((1, -1)))
        self.position_embedding = position_embedding

        # context model for embedding
        self.context_model = ContextModel(
            location_embedding.embedding_dim,
            hidden_size,
            embed_xy,
            embed_poi,
            poi_dim=poi_dim,
            device=device,
        )

        #
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads, activation="gelu", batch_first=True
        )
        encoder_norm = nn.LayerNorm(hidden_size)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

    def forward(self, src_tokens, context=None):
        x = self.forward_embedding(src_tokens, context)

        # B x T, True will be ignored
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        # decoder model
        hidden = self.model(x, src_key_padding_mask=encoder_padding_mask)

        return {
            "encoder_out": hidden,  # B x T x C
            "encoder_padding_mask": encoder_padding_mask,  # B x T
        }

    def forward_embedding(self, src_tokens, context=None):
        x = self.location_embedding(src_tokens)

        if self.duration_embedding is not None:
            x += self.duration_embedding(context["duration"].unsqueeze(-1))

        if self.mode_embedding is not None:
            x += self.mode_embedding(context["mode"])

        x = self.embed_scale * x

        # up-projection
        x = self.input_up_proj(x)

        # context
        x = self.context_model(x, context)

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
        mode_embedding=None,
        duration_embedding=None,
        position_embedding=None,
        input_up_proj=None,
        output_down_proj=None,
        self_cond=False,
    ):
        super().__init__()

        # up projection (shared)
        self.input_up_proj = input_up_proj
        self.output_down_proj = output_down_proj

        #
        self.location_embedding = location_embedding
        self.embed_scale = math.sqrt(location_embedding.embedding_dim)

        #
        self.duration_embedding = duration_embedding

        #
        self.mode_embedding = mode_embedding

        self.hidden_size = hidden_size
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        # position embeddings (shared)
        max_positions = 512
        self.register_buffer("position_ids", torch.arange(max_positions).expand((1, -1)))
        self.position_embedding = position_embedding

        self.self_cond = self_cond
        if self_cond:
            self_cond_dim = location_embedding.embedding_dim
            self.self_cond_proj = nn.Sequential(
                nn.Linear(self_cond_dim * 2, self_cond_dim),
                nn.ReLU(),
                nn.Linear(self_cond_dim, self_cond_dim),
                nn.Dropout(dropout),
            )
        #
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        #
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_attention_heads, activation="gelu", batch_first=True
        )
        decoder_norm = nn.LayerNorm(hidden_size)
        self.model = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)

    def forward_embedding(self, tgt_tokens, tgt_cxt):
        embed = self.location_embedding(tgt_tokens)

        if self.duration_embedding is not None:
            embed += self.duration_embedding(tgt_cxt["duration"].unsqueeze(-1))

        if self.mode_embedding is not None:
            embed += self.mode_embedding(tgt_cxt["mode"])

        embed = self.embed_scale * embed
        return embed

    def forward_hidden(self, z_t, t, prev_z_0_hat=None):
        seq_length = z_t.size(1)

        if self.self_cond:
            cat_embed = torch.cat((z_t, prev_z_0_hat), dim=-1)
            hidden = self.input_up_proj(self.self_cond_proj(cat_embed))
        else:
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

    def forward(self, z_t, t, padding_mask, encoder_out, prev_z_0_hat=None):
        hidden = self.forward_hidden(z_t, t, prev_z_0_hat)

        hidden = self.model(
            tgt=hidden,  # B x T x C
            memory=encoder_out["encoder_out"],  # B x T x C
            tgt_key_padding_mask=~padding_mask,
            memory_key_padding_mask=encoder_out["encoder_padding_mask"],  # B x T
        )

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

        # duration embedding
        self.if_include_duration = model_args.if_include_duration
        self.duration_embedding = None
        if self.if_include_duration:
            self.duration_embedding = nn.Linear(1, model_args.input_dims, bias=False)

        # mode embedding
        self.if_include_mode = model_args.if_include_mode
        self.mode_embedding = None
        if self.if_include_mode:
            self.mode_embedding = nn.Embedding(model_args.max_mode, model_args.input_dims, padding_idx=0)

        # up
        self.input_up_proj = nn.Linear(model_args.input_dims, model_args.hidden_size, bias=False)

        # position embedding
        self.position_embedding = nn.Embedding(max_positions, model_args.hidden_size)

        # down
        self.output_down_proj = nn.Linear(model_args.hidden_size, model_args.input_dims, bias=False)

        # encoder
        self.encoder = TransEncoder(
            num_encoder_layers=model_args.num_layers,
            hidden_size=model_args.hidden_size,
            num_attention_heads=model_args.num_attention_heads,
            dropout=model_args.dropout,
            location_embedding=self.location_embedding,
            mode_embedding=self.mode_embedding,
            duration_embedding=self.duration_embedding,
            position_embedding=self.position_embedding,
            input_up_proj=self.input_up_proj,
            embed_xy=model_args.if_embed_xy,
            embed_poi=model_args.if_embed_poi,
            poi_dim=model_args.poi_dim,
            device=model_args.device,
        )

        # decoder
        self.decoder = TransDecoder(
            num_decoder_layers=model_args.num_layers,
            hidden_size=model_args.hidden_size,
            num_attention_heads=model_args.num_attention_heads,
            dropout=model_args.dropout,
            location_embedding=self.location_embedding,
            mode_embedding=self.mode_embedding,
            duration_embedding=self.duration_embedding,
            position_embedding=self.position_embedding,
            input_up_proj=self.input_up_proj,
            output_down_proj=self.output_down_proj,
            self_cond=diff_args.self_cond,
        )

        # location head
        self.lm_head = nn.Linear(model_args.input_dims, model_args.max_location, bias=False)
        # weight sharing
        with torch.no_grad():
            self.lm_head.weight = self.location_embedding.weight

        # mode head
        if self.if_include_mode:
            self.lm_head_mode = nn.Linear(model_args.input_dims, model_args.max_mode, bias=False)
            # weight sharing
            with torch.no_grad():
                self.lm_head_mode.weight = self.mode_embedding.weight

        # duration head
        if self.if_include_duration:
            self.lm_head_duration = nn.Sequential(
                nn.Linear(model_args.input_dims, model_args.input_dims),
                nn.ReLU(),
                nn.Linear(model_args.input_dims, 1),
            )

        self.training_diffusion = GaussianDiffusion(
            betas=get_named_beta_schedule(
                diff_args.noise_schedule,
                diff_args.diffusion_steps,
                diff_args.rescaling_factor,
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
                diff_args.decoding_rescaling_factor,
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

    def get_duration_prediction(self, hidden_repr):
        return self.lm_head_duration(hidden_repr)

    def get_mode_prediction(self, hidden_repr):
        return self.lm_head_mode(hidden_repr)

    def forward(self, src, tgt, src_ctx, tgt_cxt, t):
        """Compute training losses"""

        # encoding
        encoder_out = self.encoder(src, context=src_ctx)

        # padding_mask B x T
        mask = tgt.ne(0)

        # diffusion
        z_0 = self.decoder.forward_embedding(tgt, tgt_cxt)
        model_t = t * self.timesteps_scale

        noise = torch.randn_like(z_0) * self.diff_args.rescaling_factor
        # reparametrization trick
        z_t = self.training_diffusion.q_sample(z_0, t, noise).type_as(z_0)

        # self-conditioning
        prev_z_0_hat = torch.zeros_like(z_0)
        if self.diff_args.self_cond and random() < 0.5:
            with torch.no_grad():
                prev_z_0_hat = self.decoder(z_t, model_t, mask, encoder_out, prev_z_0_hat).detach()

        # model use z_t to predict z_0
        z_0_hat = self.decoder(z_t, model_t, mask, encoder_out, prev_z_0_hat)
        logits = self.get_logits(z_0 if self.diff_args.rounding_loss else z_0_hat)

        terms = {}

        #
        terms["mse"] = mean_flat((z_0_hat - z_0).square(), mask)

        # Rounding error: embedding regularization
        terms["head_nll"] = token_discrete_loss(logits, tgt, mask=mask, label_smoothing=0.1)

        terms["loss"] = terms["mse"] + terms["head_nll"]

        if self.if_include_mode:
            mode_pred = self.get_mode_prediction(z_0)
            terms["head_mode"] = token_discrete_loss(mode_pred, tgt_cxt["mode"], mask=mask, label_smoothing=0.1)
            # terms["loss"] += 0.1 * terms["head_mode"] / (terms["head_mode"] / terms["head_nll"]).detach()
            terms["loss"] += terms["head_mode"]

        if self.if_include_duration:
            duration_pred = self.get_duration_prediction(z_0)
            terms["head_mse"] = prediction_mse_loss(duration_pred, tgt_cxt["duration"], mask=mask)
            # terms["loss"] += 0.1 * terms["head_mse"] / (terms["head_mse"] / terms["head_nll"]).detach()
            terms["loss"] += terms["head_mse"]

        return terms

    def forward_decoder(self, z_t, step, mask, encoder_out, prev_z_0_hat=None):
        """Sample z_{t-1} given z_t"""

        # rescale timesteps
        model_t = self.decoding_diffusion.timestep_map[step] * self.timesteps_scale
        model_t = torch.full([len(z_t)], model_t, device=z_t.device)

        # predict z_0
        z_0_hat = self.decoder(z_t, model_t, mask, encoder_out, prev_z_0_hat)

        # clamping trick
        if self.diff_args.clamping:
            tokens = self.get_logits(z_0_hat).argmax(-1)
            #
            ctx = {}
            ctx["duration"] = torch.clamp(self.get_duration_prediction(z_0_hat).squeeze(-1), min=-1, max=1)
            ctx["mode"] = self.get_mode_prediction(z_0_hat).argmax(-1)
            #
            z_0_hat = self.decoder.forward_embedding(tokens, tgt_cxt=ctx)

        # sample z_{t-1}
        t = torch.tensor(step, device=z_t.device)
        mean, _, log_variance = self.decoding_diffusion.q_posterior_mean_variance(z_0_hat, z_t, t)
        noise = torch.randn_like(z_t) * self.diff_args.decoding_rescaling_factor

        z_t = mean + (0.5 * log_variance).exp() * noise
        z_t = z_t.type_as(z_0_hat)

        return z_t, z_0_hat

    def forward_output_layer(self, z_t):
        scores, tokens = self.get_logits(z_t).log_softmax(-1).max(-1)

        durations = self.get_duration_prediction(z_t).squeeze(-1)
        durations = (torch.clamp(durations, min=-1, max=1) + 1) / 2 * 2880

        _, modes = self.get_mode_prediction(z_t).log_softmax(-1).max(-1)

        return tokens, durations, modes, scores

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


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


def prediction_mse_loss(pred, true_tgt, mask=None):
    loss_fct = torch.nn.MSELoss(reduction="none")
    decoder_nll = loss_fct(pred.squeeze(-1), true_tgt).view(true_tgt.shape)
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

        def alpha_bar(t):
            return 1 - np.sqrt(t + shift)

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    f2 = rescaling_factor**2

    def rescaled_alpha_bar(t):
        return alpha_bar(t) / (f2 - (f2 - 1) * alpha_bar(t))

    return betas_for_alpha_bar(num_diffusion_timesteps, rescaled_alpha_bar)
