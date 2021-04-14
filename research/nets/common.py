from define_config import env_fn
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
import torch.distributions.transforms as tran
from torch import nn
import torch.nn.functional as F

def zero_module(module):
  """Zero out the parameters of a module and return it."""
  for p in module.parameters():
    p.detach().zero_()
  return module

class CausalSelfAttention(nn.Module):
  """
  A vanilla multi-head masked self-attention layer with a projection at the end.
  It is possible to use th.nn.MultiheadAttention here but I am including an
  explicit implementation here to show that there is nothing too scary here.
  """

  def __init__(self, block_size, G):
    super().__init__()
    self.block_size = block_size
    assert G.n_embed % G.n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(G.n_embed, G.n_embed)
    self.query = nn.Linear(G.n_embed, G.n_embed)
    self.value = nn.Linear(G.n_embed, G.n_embed)
    # output projection
    self.proj = nn.Linear(G.n_embed, G.n_embed)
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", th.tril(th.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size))
    self.G = G

  def forward(self, x, layer_past=None):
    B, T, G = x.size()
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    k = self.key(x).view(B, T, self.G.n_head, G // self.G.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.G.n_head, G // self.G.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.G.n_head, G // self.G.n_head).transpose(1, 2)  # (B, nh, T, hs)
    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
    att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, G)  # re-assemble all head outputs side by side
    # output projection
    y = self.proj(y)
    return y

class TransformerBlock(nn.Module):
  """ an unassuming Transformer block (from @karpathy)"""
  def __init__(self, block_size, G):
    super().__init__()
    self.ln1 = nn.LayerNorm(G.n_embed)
    self.ln2 = nn.LayerNorm(G.n_embed)
    self.attn = CausalSelfAttention(block_size, G)
    self.mlp = nn.Sequential(
        nn.Linear(G.n_embed, 4 * G.n_embed),
        nn.GELU(),
        nn.Linear(4 * G.n_embed, G.n_embed),
    )
  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x

class GaussHead(nn.Module):
  def __init__(self, input_size, z_size, G):
    super().__init__()
    self.G = G
    self.z_size = z_size
    self.layer = nn.Linear(input_size, 2 * z_size)

  def forward(self, x, past_z=None):
    mu, log_std = self.layer(x).chunk(2, -1)
    std = F.softplus(log_std) + self.G.min_std
    if past_z is not None:
      mu = mu + past_z
    #dist = thd.Independent(thd.Normal(mu, std), 1)
    #dist = thd.Normal(mu, std)
    dist = thd.MultivariateNormal(mu, th.diag_embed(std))
    return dist

class MDNHead(nn.Module):
  def __init__(self, in_n, out_n, G):
    super().__init__()
    self.G = G
    self.out_n = out_n
    shape = self.G.mdn_k + 2 * self.out_n * self.G.mdn_k
    self.layer = nn.Linear(in_n, shape)

  def forward(self, x):
    dx = self.G.mdn_k * self.out_n
    out = self.layer(x)
    mu = out[..., :dx]
    std = F.softplus(out[..., dx:2 * dx]) + self.G.min_std
    logits = out[..., 2 * dx:]
    # TODO: should this be view or something
    mu = mu.reshape(list(mu.shape[:-1]) + [self.G.mdn_k, -1])
    std = std.reshape(list(std.shape[:-1]) + [self.G.mdn_k, -1])
    if past_o is not None:
      mu = mu + past_o[..., None, :]
    cat = thd.Categorical(logits=logits)
    dist = thd.MixtureSameFamily(cat, thd.MultivariateNormal(mu, th.diag_embed(std)))
    return dist

class CategoricalHead(nn.Module):
  """take latent and produce a multinomial distribution independently"""
  def __init__(self, in_n, out_n, G):
    super().__init__()
    self.layer = nn.Linear(in_n, out_n)
  def forward(self, x):
    x = self.layer(x)
    return thd.Multinomial(logits=x)

class BinaryHead(nn.Module):
  """take latent and produce a bernoulli distribution"""
  def __init__(self, in_n, out_n, G):
    super().__init__()
    self.layer = nn.Linear(in_n, out_n)
  def forward(self, x):
    x = self.layer(x)
    return thd.Bernoulli(logits=x)

class ConvBinHead(nn.Module):
  def __init__(self, in_n, out_n, G):
    super().__init__()
    self.G = G
    self.in_n = in_n
    self.out_n = out_n
    #self.d1 = nn.ConvTranspose2d(self.in_n//self.shape, 64, 7, stride=2)
    #self.d2 = nn.ConvTranspose2d(64, 1, 4, stride=2)
    first_kernel = int(self.G.wh_ratio * 4)
    self.net = nn.Sequential(
      nn.ConvTranspose2d(self.in_n, 64, (4,first_kernel), stride=2),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
    )

  def forward(self, x):
    BS, LEN, G = x.shape
    x = x.reshape(BS * LEN, G, 1, 1)
    x = self.net(x)
    x = x.reshape(BS, LEN, -1)
    return thd.Bernoulli(logits=x)

class ConvEmbed(nn.Module):
  def __init__(self, in_n, out_n, G):
    super().__init__()
    self.G = G
    self.c1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
    self.c2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

  def forward(self, x):
    BS, LEN, G = x.shape
    x = x.reshape(BS * LEN, -1, self.G.lcd_h, self.G.lcd_w)
    x = self.c1(x)
    x = F.relu(x)
    x = self.c2(x)
    x = x.reshape(BS, LEN, -1)
    return x

class GPTDist(nn.Module):
  def __init__(self, gpt, x):
    super().__init__()
    self.gpt = gpt
    self.x = x
  def log_prob(self, state):
    x = state.flatten(0,1)[...,None]
    dist = self.gpt.forward(x, cond=self.x.flatten(0,1))
    return dist.log_prob(x)
  def sample(self, prompts=None):
    shape = self.x.shape
    return self.gpt.sample(np.prod(shape[:2]), cond=self.x.flatten(0,1), prompts=prompts)[0].reshape(shape[:2] + (-1,))

class MultiHead(nn.Module):
  def __init__(self, in_n, out_n, split, G):
    super().__init__()
    self.G = G
    self.in_n = in_n
    self.out_n = out_n
    self.split = split
    self.layer = nn.Linear(in_n, in_n * 2)
    if self.G.conv_io:
      self.binary = ConvBinHead(in_n, self.split, G)
    else:
      self.binary = BinaryHead(in_n, self.split, G)
    #self.state = FlowHead(in_n, out_n - self.split, G)
    #self.state = GaussHead(in_n, out_n-self.split, G) 
    self.state = MDNHead(in_n, out_n-self.split, G) 
    #self.state = GPT(1, block_size=out_n-self.split, dist='mdn', cond=in_n, G=G) 

  def forward(self, x):
    xb, xs = self.layer(x).chunk(2, -1)
    bin = self.binary(xb) 
    state = self.state(xs)
    return {'lcd': bin, 'pstate': state}

class ResBlock(nn.Module):
  def __init__(self, channels, emb_channels, out_channels=None, dropout=0.0, group_size=16):
    super().__init__()
    self.out_channels = out_channels or channels

    self.in_layers = nn.Sequential(
        nn.GroupNorm(group_size, channels),
        nn.SiLU(),
        nn.Conv2d(channels, self.out_channels, 3, padding=1)
    )
    self.emb_layers = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_channels, self.out_channels)
    )
    self.out_layers = nn.Sequential(
        nn.GroupNorm(group_size, self.out_channels),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
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


def aggregate(x, dim=1, catdim=-1):
  """
  https://arxiv.org/pdf/2004.05718.pdf

  takes (BS, N, E). extra args change the axis of N

  returns (BS, 4E) where 4E is min, max, std, mean aggregations.
                   using all of these rather than just one leads to better coverage. see paper
  """
  min = th.min(x, dim=dim)[0]
  max = th.max(x, dim=dim)[0]
  std = th.std(x, dim=dim)
  mean = th.mean(x, dim=dim)
  return th.cat([min, max, std, mean], dim=catdim)

