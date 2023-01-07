import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from research.nets.common import CrossAttentionBlock, SelfAttentionBlock, zero_module


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
        for i in range(self.num_layers):
            z = self.layers[i](q=z)
        x = self.output_cross(q=x, kv=z)
        return z, x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class InterfaceNet(nn.Module):
    def __init__(self, resolution, hidden_size):
        super().__init__()
        assert resolution == 64
        patch_size = 4
        n_z = 128
        dim_z = 256

        num_blocks = 3
        num_layers = 3
        sizex = resolution // patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_size * patch_size * 3, dim_z),
            nn.LayerNorm(dim_z),
        )
        Emb = lambda *size: nn.Parameter(
            nn.init.trunc_normal_(torch.zeros(*size), std=0.02)
        )

        self.pos_emb = Emb(1, sizex * sizex, dim_z)
        self.skip_z = nn.Sequential(
            nn.Linear(dim_z, 4 * dim_z),
            nn.GELU(),
            nn.Linear(4 * dim_z, dim_z),
            zero_module(nn.LayerNorm(dim_z)),
        )

        self.z_emb = Emb(n_z, dim_z)

        self.block = Block(sizex * sizex, dim_z, 4, num_layers)
        # self.blocks = nn.Sequential(
        #    *[Block(sizex*sizex, dim_z, 4, num_layers) for _ in range(num_blocks)]
        # )

        self.proj_out = nn.Sequential(
            nn.LayerNorm(dim_z),
            nn.Linear(dim_z, 3 * patch_size * patch_size),
            Rearrange(
                'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                h=sizex,
                w=sizex,
                p1=patch_size,
                p2=patch_size,
            ),
        )

    def forward(self, x, logsnr, prev_z, *args, **kwargs):
        # TODO: does logsnr just become an extra token. i think so..
        x = self.to_patch_embedding(x) + self.pos_emb

        # tokenize image x into patches
        z = self.z_emb + self.skip_z(prev_z)

        z, x = self.block(z, x)

        x = self.proj_out(x)
        return x, z
