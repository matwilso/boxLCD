import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from research.nets.common import CrossAttentionBlock, SelfAttentionBlock, zero_module, Residual, timestep_embedding


class Block(nn.Module):
    def __init__(self, block_size, n_embed, n_head, num_layers):
        super().__init__()
        self.input_cross = CrossAttentionBlock(block_size, n_embed, n_head)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(SelfAttentionBlock(block_size, n_embed, n_head))
        self.output_cross = CrossAttentionBlock(block_size, n_embed, n_head)
        self.num_layers = num_layers

    def forward(self, z, x):
        z = self.input_cross(q=z, kv=x)
        for layer in self.layers:
            z = layer(q=z)
        x = self.output_cross(q=x, kv=z)
        return z, x


class VideoInterfaceNet(nn.Module):
    def __init__(self, resolution, temp_res, G):
        super().__init__()
        #assert resolution == 64
        self.patch_t = G.patch_t
        self.patch_h = G.patch_h
        self.patch_w = G.patch_w

        self.n_z = G.n_z
        self.dim_z = G.dim_z
        num_blocks = G.num_blocks
        num_layers = G.num_layers
        num_head = G.n_head

        size_t = temp_res // self.patch_t
        size_h = resolution // self.patch_h
        size_w = resolution // self.patch_w

        comb_patch = self.patch_t * self.patch_h * self.patch_w
        comb_size = size_t * size_h * size_w

        self.patch_in = nn.Sequential(
            Rearrange('b c (t pt) (h ph) (w pw) -> b (h w t) (pt ph pw c)', pt=self.patch_t, ph=self.patch_h, pw=self.patch_w),
            nn.Linear(comb_patch * 3, self.dim_z),
            nn.LayerNorm(self.dim_z),
        )
        Emb = lambda *size: nn.Parameter(
            nn.init.trunc_normal_(torch.zeros(*size), std=0.02)
        )

        self.pos_emb = Emb(1, comb_size, self.dim_z)
        self.skip_z = nn.Sequential(
            nn.Linear(self.dim_z, 4 * self.dim_z),
            nn.GELU(),
            nn.Linear(4 * self.dim_z, self.dim_z),
            zero_module(nn.LayerNorm(self.dim_z)),
        )

        self.z_emb = Emb(1, self.n_z + 1, self.dim_z)

        self.blocks = nn.ModuleList([Block(comb_size, self.dim_z, num_head, num_layers) for _ in range(num_blocks)])

        self.patch_out = nn.Sequential(
            nn.LayerNorm(self.dim_z),
            nn.Linear(self.dim_z, 3 * comb_patch),
            Rearrange(
                'b (h w t) (pt ph pw c) -> b c (t pt) (h ph) (w pw)',
                t=size_t,
                h=size_h,
                w=size_w,
                c=3,
                pt=self.patch_t,
                ph=self.patch_h,
                pw=self.patch_w,
            ),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(64, 4*self.dim_z),
            nn.GELU(),
            nn.Linear(4*self.dim_z, self.dim_z),
        )
    
    def train_fwd(self, x, logsnr, self_cond, *args, **kwargs):
        prev_lz = torch.zeros((x.shape[0], self.n_z, self.dim_z), device=x.device)

        if self_cond:
            with torch.no_grad():
                # run the net first with 0 latent code, but actually train on the final output
                _, prev_lz = self.forward(x, logsnr, prev_lz, *args, **kwargs)

        x, _ = self.forward(x, logsnr, prev_lz, *args, **kwargs)
        return {'x': x}

    def sample_fwd(self, x, logsnr, *args, **kwargs):
        prev_lz = kwargs.get('lz')
        if prev_lz is None:
            prev_lz = torch.zeros((x.shape[0], self.n_z, self.dim_z), device=x.device)
        x, lz = self.forward(x, logsnr, prev_lz, *args, **kwargs)
        return {'x': x, 'lz': lz}

    def forward(self, x, logsnr, prev_lz, *args, **kwargs):
        # tokenize image x into patches
        x = self.patch_in(x) + self.pos_emb

        # include timestep as a token
        time_emb = self.time_embed(timestep_embedding(timesteps=logsnr, dim=64, max_period=256))
        if prev_lz.shape[1] == self.n_z:
            prev_lz = torch.cat([prev_lz, time_emb[:, None]], dim=1)
        else:
            prev_lz[:, -1] = time_emb

        z = self.z_emb + self.skip_z(prev_lz)
        for block in self.blocks:
            z, x = block(z, x)
        x = self.patch_out(x)
        return x, z