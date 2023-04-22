# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch

from model import monotonic_align
from model.base import BaseModule
from model.diffusion import Diffusion, Diffusion_Motion
from model.text_encoder import MuMotionEncoder, TextEncoder
from model.utils import (duration_loss, fix_len_compatibility, generate_path,
                         sequence_mask)


class GradTTS(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, n_motions, dec_dim, beta_min, beta_max, pe_scale, 
                 mu_motion_encoder_params, decoder_motion_type, motion_reduction_factor, only_speech=False, encoder_type="default"):
        super(GradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.n_motions = n_motions
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.generate_motion = not only_speech
        self.motion_reduction_factor = motion_reduction_factor

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size, encoder_type=encoder_type)
        self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)
        
        if self.generate_motion: 
            # This is to load old checkpoints
            if 'prior_loss' in mu_motion_encoder_params:
                self.motion_prior_loss = mu_motion_encoder_params.pop('prior_loss')
            else:
                self.motion_prior_loss = False
            # Remove once all checkpoints have prior loss as a parameter

            self.mu_motion_encoder = MuMotionEncoder(
                input_channels=n_feats,
                output_channels=n_motions,
                **mu_motion_encoder_params
            )
            if decoder_motion_type == 'wavegrad': 
                self.decoder_motion = Diffusion_Motion(
                    in_channels=n_motions,
                    beta_min=beta_min,
                    beta_max=beta_max,
                )
            elif decoder_motion_type == 'gradtts': 
                self.decoder_motion = Diffusion(
                    n_feats=n_motions,
                    dim = dec_dim,
                    n_spks=n_spks,
                    spk_emb_dim=spk_emb_dim,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    pe_scale=pe_scale
                )

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])
        
        if isinstance(n_timesteps, dict):
            n_timestep_mel = n_timesteps['mel']
            n_timestep_motion = n_timesteps['motion']
        else:
            n_timestep_mel = n_timesteps
            n_timestep_motion = n_timesteps
        
        if isinstance(temperature, dict):
            temperature_mel = temperature['mel']
            temperature_motion = temperature['motion']
        else:
            temperature_mel = temperature
            temperature_motion = temperature
            

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]
        
        
        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature_mel
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timestep_mel, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        if self.generate_motion:
            mu_y_motion = mu_y[:, :, ::self.motion_reduction_factor] 
            y_motion_mask = y_mask[:, :, ::self.motion_reduction_factor]
            mu_y_motion = self.mu_motion_encoder(mu_y_motion, y_motion_mask)
            encoder_outputs_motion = mu_y_motion[:, :, :y_max_length]
            # sample latent representation from terminal distribution N(mu_y_motion, I)
            z_motion = mu_y_motion + torch.randn_like(mu_y_motion, device=mu_y_motion.device) / temperature_motion 
            # Generate sample by performing reverse dynamics
            decoder_outputs_motion = self.decoder_motion(z_motion, y_motion_mask, mu_y_motion, n_timestep_motion, stoc, spk)
            decoder_outputs_motion = decoder_outputs_motion[:, :, :y_max_length]
        else:
            decoder_outputs_motion = None
            encoder_outputs_motion = None

        return encoder_outputs, decoder_outputs, encoder_outputs_motion,  decoder_outputs_motion, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, y_motion, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad(): 
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)    # cut a random segment of size `out_size` from each sample in batch max_offset: [758, 160, 773]
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))  # offset ranges for each sample in batch offset_ranges: [(0, 758), (0, 160), (0, 773)]
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            if self.generate_motion:
                y_motion_cut = torch.zeros(y_motion.shape[0], self.n_motions, out_size, dtype=y_motion.dtype, device=y_motion.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                if self.generate_motion:
                    y_motion_cut[i, :, :y_cut_length] = y_motion[i, :, cut_lower:cut_upper]

                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            
            attn = attn_cut
            y = y_cut
            if self.generate_motion:
                y_motion = y_motion_cut

            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        
        

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)
        if self.generate_motion:
            # Reduce motion features
            mu_y_motion = mu_y[:, :, ::self.motion_reduction_factor]
            y_motion_mask = y_mask[:, :, ::self.motion_reduction_factor]
            y_motion = y_motion[:, :, ::self.motion_reduction_factor]
            
            mu_y_motion = self.mu_motion_encoder(mu_y_motion, y_motion_mask)
            diff_loss_motion, xt_motion = self.decoder_motion.compute_loss(y_motion, y_motion_mask, mu_y_motion, spk)
        else:
            diff_loss_motion = 0
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        if self.generate_motion and self.motion_prior_loss:
            prior_loss_motion = torch.sum(0.5 * ((y_motion - mu_y_motion) ** 2 + math.log(2 * math.pi)) * y_motion_mask)
            prior_loss_motion = prior_loss_motion / (torch.sum(y_motion_mask) * self.n_motions)
        else:
            prior_loss_motion = 0
        
        return dur_loss, prior_loss + prior_loss_motion, diff_loss + diff_loss_motion
