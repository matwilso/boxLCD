import math

import torch
import torch.nn.functional as F
from einops import parse_shape, rearrange, reduce, repeat
from torch import nn

from research.nets.autoencoders.diffusion_v2.simple_unet import (
    TimestepEmbedSequential,
    timestep_embedding,
)
from research.nets.common import ResBlock3d, TransformerBlock

# arch maintains same shape, has resnet skips, and injects the time embedding in many places

"""
This is a shorter and simpler Unet, designed to work on MNIST.
"""

MAX_TIMESTEPS = 256


class SimpleUnet3D(nn.Module):
    def __init__(self, temporal_res, spatial_res, channels, dropout, superres=False, supertemp=False):
        super().__init__()
        out_channels = 3
        time_embed_dim = 2 * channels
        # if super-res, condition on low-res image
        in_channels = 6 if superres or supertemp else 3

        self.time_embed = nn.Sequential(
            nn.Linear(64, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.cond_w_embed = nn.Sequential(
            nn.Linear(64, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.guide_embed = nn.Sequential(
            nn.Linear(128, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.down = Down(
            temporal_res,
            spatial_res,
            in_channels,
            channels,
            time_embed_dim,
            dropout=dropout,
        )
        self.turn = ResBlock3d(channels, time_embed_dim, dropout=dropout)
        self.up = Up(temporal_res, spatial_res, channels, time_embed_dim, dropout=dropout)
        self.out = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv3d(channels, out_channels, 3, padding=1),
        )

    def set_attn_masks(self, iso_image=False):
        pass

    def forward(self, x, timesteps, guide=None, cond_w=None, low_res=None, coarse_tween=None):
        # TODO: maybe have low_res/coarse_tween catted before this function

        assert timesteps.max() < MAX_TIMESTEPS
        emb = self.time_embed(
            timestep_embedding(
                timesteps=timesteps.float(), dim=64, max_period=MAX_TIMESTEPS
            )
        )

        if guide is not None:
            guide_emb = self.guide_embed(guide)
            emb += guide_emb

        if cond_w is not None:
            cond_w_embed = self.cond_w_embed(
                timestep_embedding(timesteps=cond_w, dim=64, max_period=4)
            )
            emb += cond_w_embed

        if low_res is not None:
            x = torch.cat([x, low_res], dim=1)

        if coarse_tween is not None:
            x = torch.cat([x, coarse_tween], dim=1)

        # <UNET> downsample, then upsample with skip connections between the down and up.
        x, cache = self.down(x, emb)
        x = self.turn(x, emb)
        x = self.up(x, emb, cache)
        x = self.out(x)
        # </UNET>
        return x


class Downsample(nn.Module):
    """halve the size of the input"""

    def __init__(self, channels, out_channels=None, stride=2):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv3d(channels, out_channels, 3, stride=stride, padding=1)

    def forward(self, x, emb=None):
        return self.conv(x)


class Down(nn.Module):
    def __init__(
        self, temporal_res, spatial_res, in_channels, channels, emb_channels, dropout=0.0
    ):
        super().__init__()
        seq = [
            # not really a downsample, just makes the code simpler to reuse
            Downsample(in_channels, channels, 1),
            ResBlock3d(channels, emb_channels, dropout=dropout),
            TimestepEmbedSequential(
                ResBlock3d(channels, emb_channels, channels, dropout=dropout),
                AttentionBlock(temporal_res, channels),
            ),
            Downsample(channels),
            TimestepEmbedSequential(
                ResBlock3d(channels, emb_channels, channels, dropout=dropout),
                AttentionBlock(temporal_res // 2, channels),
            ),
            ResBlock3d(channels, emb_channels, dropout=dropout),
            Downsample(channels),
        ]
        # TODO: also add extra layers for temporal res above baseline

        extra_res = (spatial_res // 16) // 2
        for _ in range(extra_res):
            extra = [
                TimestepEmbedSequential(
                    ResBlock3d(channels, emb_channels, dropout=dropout),
                    Downsample(channels),
                )
            ]
            seq += extra

        self.seq = nn.ModuleList(seq)

    def forward(self, x, emb):
        cache = []
        for layer in self.seq:
            x = layer(x, emb)
            cache += [x]
        return x, cache


class Upsample(nn.Module):
    """double the size of the input"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x, emb=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, temporal_res, spatial_res, channels, emb_channels, dropout=0.0):
        super().__init__()
        # on the up, bundle resnets with upsampling so upsamplnig can be simpler
        seq = [
            TimestepEmbedSequential(
                ResBlock3d(2 * channels, emb_channels, channels, dropout=dropout),
                Upsample(channels),
            ),
            ResBlock3d(2 * channels, emb_channels, channels, dropout=dropout),
            ResBlock3d(2 * channels, emb_channels, channels, dropout=dropout),
            TimestepEmbedSequential(
                ResBlock3d(2 * channels, emb_channels, channels),
                Upsample(channels),
                AttentionBlock(temporal_res, channels),
            ),
            ResBlock3d(2 * channels, emb_channels, channels, dropout=dropout),
            ResBlock3d(2 * channels, emb_channels, channels, dropout=dropout),
            ResBlock3d(2 * channels, emb_channels, channels, dropout=dropout),
        ]
        extra_res = (spatial_res // 16) // 2
        for _ in range(extra_res):
            extra = [
                TimestepEmbedSequential(
                    ResBlock3d(2 * channels, emb_channels, channels, dropout=dropout),
                    Upsample(channels),
                ),
            ]
            seq = extra + seq
        self.seq = nn.ModuleList(seq)

    def forward(self, x, emb, cache):
        cache = cache[::-1]
        for i in range(len(self.seq)):
            layer, hoz_skip = self.seq[i], cache[i]
            x = torch.cat([x, hoz_skip], 1)
            x = layer(x, emb)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, n, n_embed):
        super().__init__()
        n_head = n_embed // 8
        self.attn = TransformerBlock(n, n_embed, n_head, causal=False)
        # self.attn = SelfAttention(n, n_embed, n_head, causal=False)
        self.register_buffer(
            "time_embed",
            timestep_embedding(
                timesteps=torch.linspace(0, 1, n), dim=n_embed, max_period=1
            ).T,
        )

    def forward(self, x, emb=None):
        x = x + self.time_embed[None, :, :, None, None]
        x_shape = parse_shape(x, 'bs c t h w')
        x = rearrange(x, 'bs c t h w -> (bs h w) t c')
        x = self.attn(x)
        x = rearrange(x, '(bs h w) t c -> bs c t h w', **x_shape)
        return x
