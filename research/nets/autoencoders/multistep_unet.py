from collections import namedtuple
from re import I
import numpy as np
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from research import utils
from research.nets.common import ResBlock
from torch.optim import Adam
from ._base import Autoencoder, SingleStepAE, MultiStepAE
from jax.tree_util import tree_map
import math
import torch as th
from torch import nn
import torch.nn.functional as F

# TODO: rethink groupnorm
# TODO: rethink not downsampling time dim

class DiffusionUnet3d(MultiStepAE):
  def __init__(self, env, G):
    super().__init__(env, G)
    self.z_size = 256
    state_n = env.observation_space.spaces['proprio'].shape[0]
    act_n = env.action_space.shape[0]
    G.dropout = 0.0
    self.unet = SimpleUnet3d(act_n, state_n, self.z_size, G)
    self._init()

  def _unprompted_eval(self, epoch=None, writer=None, metrics=None, batch=None, arbiter=None):
    return {}

  def _unroll_lcd(self, batch):
    lcd = batch['lcd']
    lcd = lcd[:, None]
    return lcd

  def loss(self, batch):
    lcd = self._unroll_lcd(batch)
    out = th.sigmoid(self.unet(lcd))
    recon_losses = {}
    recon_losses['loss/recon_lcd'] = (out - lcd).pow(2).mean()
    recon_loss = sum(recon_losses.values())
    metrics = {'loss/recon_total': recon_loss, **recon_losses}
    return recon_loss, metrics

  def encode(self, batch, flatten=None):
    return self._unroll_lcd(batch)

  def _decode(self, z):
    X = namedtuple('X', 'probs')
    return {'lcd': X(th.sigmoid(self.unet(z))[:, 0])}

"""
# arch maintains same shape, has resnet skips, and injects the time embedding in many places

This is a shorter and simpler Unet, designed to work on MNIST.

It performs slightly worse than the one from Ho/Nichol+Dhariwal.
This is likely due to fewer layers and not using attention.
"""

class SimpleUnet3d(nn.Module):
  def __init__(self, act_n, state_n, in_size, G):
    super().__init__()
    channels = G.hidden_size
    dropout = G.dropout
    time_embed_dim = 2 * channels
    self.time_embed = nn.Sequential(
        nn.Linear(64, time_embed_dim),
        nn.SiLU(),
        nn.Linear(time_embed_dim, time_embed_dim),
    )
    self.down = Down(channels, time_embed_dim, dropout=dropout)
    self.turn = ResBlock(channels, time_embed_dim, dropout=dropout)
    self.up = Up(channels, time_embed_dim, dropout=dropout)
    self.out = nn.Sequential(
      nn.GroupNorm(32, channels),
      nn.SiLU(),
      nn.Conv3d(channels, 1, (3, 3, 3), padding=(1, 1, 1)),
      #nn.Conv3d(channels, 2, (3, 3, 3), padding=(1, 1, 1)),
    )
    self.G = G

  def forward(self, x, timesteps=None):
    timesteps = th.arange(x.shape[0], device=x.device)
    emb = self.time_embed(timestep_embedding(timesteps.float(), 64, timesteps.shape[0]))
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
    # TODO: change to no mixing across time
    self.conv = nn.Conv3d(channels, out_channels, kernel_size=(3, 3, 3), stride=(1, stride, stride), padding=(1, 1, 1))
    #self.conv = nn.Conv3d(channels, out_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1))

  def forward(self, x, emb=None):
    return self.conv(x)

class Down(nn.Module):
  def __init__(self, channels, emb_channels, dropout=0.0):
    super().__init__()
    self.seq = nn.ModuleList([
        Downsample(1, channels, 1),  # not really a downsample, just makes the code simpler to share
        ResBlock(channels, emb_channels, dropout=dropout),
        ResBlock(channels, emb_channels, dropout=dropout),
        Downsample(channels),
        ResBlock(channels, emb_channels, dropout=dropout),
        ResBlock(channels, emb_channels, dropout=dropout),
        Downsample(channels),
    ])
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
    self.conv = nn.Conv3d(channels, channels, (3, 3, 3), padding=(1, 1, 1))

  def forward(self, x, emb=None):
    x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
    x = self.conv(x)
    return x

class Up(nn.Module):
  def __init__(self, channels, emb_channels, dropout=0.0):
    super().__init__()
    # on the up, bundle resnets with upsampling so upsampling can be simpler
    self.seq = nn.ModuleList([
        TimestepEmbedSequential(ResBlock(2*channels, emb_channels, channels, dropout=dropout), Upsample(channels)),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        TimestepEmbedSequential(ResBlock(2*channels, emb_channels, channels), Upsample(channels)),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
    ])
  def forward(self, x, emb, cache):
    cache = cache[::-1]
    for i in range(len(self.seq)):
      layer, hoz_skip = self.seq[i], cache[i]
      x = th.cat([x, hoz_skip], 1)
      x = layer(x, emb)
    return x

class ResBlock(nn.Module):
  def __init__(self, channels, emb_channels, out_channels=None, dropout=0.0):
    super().__init__()
    self.out_channels = out_channels or channels

    # TODO: check out group norm. it's a bit sus
    self.in_layers = nn.Sequential(
        nn.GroupNorm(32, channels),
        nn.SiLU(),
        nn.Conv3d(channels, self.out_channels, (3, 3, 3), padding=(1, 1, 1))
    )
    self.emb_layers = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_channels, self.out_channels)
    )
    self.out_layers = nn.Sequential(
        nn.GroupNorm(32, self.out_channels),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        zero_module(nn.Conv3d(self.out_channels, self.out_channels, (3, 3, 3), padding=(1, 1, 1)))
    )
    if self.out_channels == channels:
      self.skip_connection = nn.Identity()
    else:
      self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) # step down size

  def forward(self, x, emb):
    h = self.in_layers(x)
    emb_out = self.emb_layers(emb)[..., None, None, None]
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

def timestep_embedding(timesteps, dim, max_period):
  """
  Create sinusoidal timestep embeddings.

  :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
  :param dim: the dimension of the output.
  :param max_period: controls the minimum frequency of the embeddings.
  :return: an [N x dim] Tensor of positional embeddings.
  """
  half = dim // 2
  freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
  args = timesteps[:, None].float() * freqs[None]
  embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
  if dim % 2:
    embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
  return embedding