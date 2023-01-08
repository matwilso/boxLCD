import math

import torch
import torch.nn.functional as F
from torch import nn

from research.nets.common import timestep_embedding

# arch maintains same shape, has resnet skips, and injects the time embedding in many places

"""
This is a shorter and simpler Unet, designed to work on MNIST.
"""

MAX_TIMESTEPS = 256


class SimpleUnet(nn.Module):
    def __init__(self, resolution, channels, dropout, superres=False):
        super().__init__()
        out_channels = 3
        time_embed_dim = 2 * channels
        # if super-res, condition on low-res image
        in_channels = 6 if superres else 3

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
            resolution, in_channels, channels, time_embed_dim, dropout=dropout
        )
        self.turn = ResBlock(channels, time_embed_dim, dropout=dropout)
        self.up = Up(resolution, channels, time_embed_dim, dropout=dropout)
        self.out = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, guide=None, cond_w=None, low_res=None):
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
        self.conv = nn.Conv2d(channels, out_channels, 3, stride=stride, padding=1)

    def forward(self, x, emb=None):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, resolution, in_channels, channels, emb_channels, dropout=0.0):
        super().__init__()
        seq = [
            # not really a downsample, just makes the code simpler to reuse
            Downsample(in_channels, channels, 1),
            ResBlock(channels, emb_channels, dropout=dropout),
            ResBlock(channels, emb_channels, dropout=dropout),
            Downsample(channels),
            ResBlock(channels, emb_channels, dropout=dropout),
            ResBlock(channels, emb_channels, dropout=dropout),
            Downsample(channels),
        ]
        extra_res = (resolution // 16) // 2
        for _ in range(extra_res):
            extra = [
                TimestepEmbedSequential(
                    ResBlock(channels, emb_channels, dropout=dropout),
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
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, emb=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, resolution, channels, emb_channels, dropout=0.0):
        super().__init__()
        # on the up, bundle resnets with upsampling so upsamplnig can be simpler
        seq = [
            TimestepEmbedSequential(
                ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
                Upsample(channels),
            ),
            ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
            ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
            TimestepEmbedSequential(
                ResBlock(2 * channels, emb_channels, channels), Upsample(channels)
            ),
            ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
            ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
            ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
        ]
        extra_res = (resolution // 16) // 2
        for _ in range(extra_res):
            extra = [
                TimestepEmbedSequential(
                    ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
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


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, self.out_channels)
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 1
            )  # step down size

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)[..., None, None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class TimestepEmbedSequential(nn.Sequential):
    """just a sequential that enables you to pass in emb also"""

    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb)
        return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module
