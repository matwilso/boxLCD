from os import stat
import sys
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import torch as torchvision
from torch.optim import Adam
from itertools import chain, count
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead, aggregate, MultiHead, ConvEmbed
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vqvae import VectorQuantizer
import utils

class MultiEnc(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    # encoder -> VQ -> decoder
    self.encoder = Encoder(env, C)
    self.vq = VectorQuantizer(C.vqK, C.vqD, C.beta, C)
    self.decoder = Decoder(env, C)

  def loss(self, batch, eval=False, return_idxs=False):
    embed_loss, decoded, perplexity, idxs = self.forward(batch)
    recon_losses = {}
    for key in decoded:
      recon_losses['recon_'+key] = -decoded[key].log_prob(batch[key]).mean()
    recon_loss = sum(recon_losses.values())
    loss = recon_loss + embed_loss
    metrics = {'vq_vae_loss': loss, 'embed_loss': embed_loss, 'perplexity': perplexity, **recon_losses, 'recon_loss': recon_loss}
    if eval: metrics['decoded'] = decoded
    if return_idxs: metrics['idxs'] = idxs
    return loss, metrics

  def forward(self, x):
    z_e = self.encoder(x)
    embed_loss, z_q, perplexity, idxs = self.vq(z_e)
    decoded = self.decoder(z_q)
    return embed_loss, decoded, perplexity, idxs

class Encoder(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    state_n = env.observation_space.spaces['pstate'].shape[0]
    act_n = env.action_space.shape[0]

    self.state_embed = nn.Sequential(
      nn.Linear(state_n+act_n, H),
      nn.ReLU(),
      nn.Linear(H, H//C.vidstack),
      nn.ReLU(),
      nn.Flatten(-2),
      nn.Linear(H, H),
    )
    self.seq = nn.ModuleList([
        nn.Conv2d(C.vidstack, H, 3, 1, padding=1),
        ResBlock(H, emb_channels=H),
        nn.Conv2d(H, H, 3, 2, padding=1),
        ResBlock(H, emb_channels=H),
        nn.Conv2d(H, H, 3, 2, padding=1),
        ResBlock(H, emb_channels=H),
    ])

  def forward(self, batch):
    acts = batch['acts']
    state = batch['pstate']
    lcd = batch['lcd']
    emb = self.state_embed(th.cat([state, acts],-1))
    x = lcd
    for layer in self.seq:
      if isinstance(layer, ResBlock):
        x = layer(x, emb)
      else:
        x = layer(x)
    return x

class Upsample(nn.Module):
  """double the size of the input"""
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
  def forward(self, x, emb=None):
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = self.conv(x)
    return x

class Decoder(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    state_n = env.observation_space.spaces['pstate'].shape[0]
    self.state_net = nn.Sequential(
      nn.Flatten(-3),
      nn.Linear(C.vqD*4*6, H*C.vidstack),
      nn.Unflatten(-1, (C.vidstack, H)),
      nn.ReLU(),
      nn.Linear(H, H),
      nn.ReLU(),
      nn.Linear(H, state_n),
    )

    self.net = nn.Sequential(
        Upsample(C.vqD, H),
        nn.ReLU(),
        Upsample(H, H),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, C.vidstack, 3, padding=1),
    )
  def forward(self, x):
    lcd_dist = thd.Bernoulli(logits=self.net(x))
    state_dist = thd.Normal(self.state_net(x), 1)
    return {'lcd': lcd_dist, 'pstate': state_dist}

class ResBlock(nn.Module):
  def __init__(self, channels, emb_channels, out_channels=None, dropout=0.0):
    super().__init__()
    self.out_channels = out_channels or channels

    self.in_layers = nn.Sequential(
        nn.GroupNorm(32, channels),
        nn.SiLU(),
        nn.Conv2d(channels, self.out_channels, 3, padding=1)
    )
    self.emb_layers = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_channels, self.out_channels)
    )
    self.out_layers = nn.Sequential(
        nn.GroupNorm(32, self.out_channels),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        utils.zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
    )
    if self.out_channels == channels:
      self.skip_connection = nn.Identity()
    else:
      self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)  # step down size

  def forward(self, x, emb):
    h = self.in_layers(x)
    emb_out = self.emb_layers(emb)[..., None, None]
    h = h + emb_out
    h = self.out_layers(h)
    return self.skip_connection(x) + h
