import sys
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F

def aggregate(x, dim=1, catdim=-1):
  """
  https://arxiv.org/pdf/2004.05718.pdf

  takes (BS, N, E). extra args change the axis of N

  returns (BS, 4E) where 4E is min, max, std, mean aggregations.
                   using all of these rather than just one leads to better coverage. see paper
  """
  min = torch.min(x, dim=dim)[0]
  max = torch.max(x, dim=dim)[0]
  std = torch.std(x, dim=dim)
  mean = torch.mean(x, dim=dim)
  return torch.cat([min, max, std, mean], dim=catdim)


class CausalSelfAttention(nn.Module):
  """
  A vanilla multi-head masked self-attention layer with a projection at the end.
  It is possible to use torch.nn.MultiheadAttention here but I am including an
  explicit implementation here to show that there is nothing too scary here.
  """

  def __init__(self, block_size, C):
    super().__init__()
    self.block_size = block_size
    assert C.n_embed % C.n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(C.n_embed, C.n_embed)
    self.query = nn.Linear(C.n_embed, C.n_embed)
    self.value = nn.Linear(C.n_embed, C.n_embed)
    # output projection
    self.proj = nn.Linear(C.n_embed, C.n_embed)
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size))
    self.C = C

  def forward(self, x, layer_past=None):
    B, T, C = x.size()
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    k = self.key(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
    att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
    # output projection
    y = self.proj(y)
    return y

class Block(nn.Module):
  """ an unassuming Transformer block """

  def __init__(self, block_size, C):
    super().__init__()
    self.ln1 = nn.LayerNorm(C.n_embed)
    self.ln2 = nn.LayerNorm(C.n_embed)
    self.attn = CausalSelfAttention(block_size, C)
    self.mlp = nn.Sequential(
        nn.Linear(C.n_embed, 4 * C.n_embed),
        nn.GELU(),
        nn.Linear(4 * C.n_embed, C.n_embed),
    )

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x

class GaussHead(nn.Module):
  def __init__(self, input_size, z_size, C):
    super().__init__()
    self.C = C
    self.z_size = z_size
    self.layer = nn.Linear(input_size, 2 * z_size)

  def forward(self, x, past_z=None):
    mu, log_std = self.layer(x).chunk(2, -1)
    std = F.softplus(log_std) + self.C.min_std
    if past_z is not None:
      mu = mu + past_z
    #dist = tdib.Independent(tdib.Normal(mu, std), 1)
    #dist = tdib.Normal(mu, std)
    dist = tdib.MultivariateNormal(mu, torch.diag_embed(std))
    return dist

class MDNHead(nn.Module):
  def __init__(self, obs_n, C):
    super().__init__()
    self.C = C
    self.obs_n = obs_n
    shape = self.C.mdn_k + 2 * self.obs_n * self.C.mdn_k
    self.layer = nn.Linear(C.n_embed, shape)

  def forward(self, x, past_o=None):
    dx = self.C.mdn_k * self.obs_n
    out = self.layer(x)
    mu = out[..., :dx]
    std = F.softplus(out[..., dx:2 * dx]) + self.C.min_std
    logits = out[..., 2 * dx:]
    # TODO: should this be view or something
    mu = mu.reshape(list(mu.shape[:-1]) + [self.C.mdn_k, -1])
    std = std.reshape(list(std.shape[:-1]) + [self.C.mdn_k, -1])
    if past_o is not None:
      mu = mu + past_o[..., None, :]
    cat = tdib.Categorical(logits=logits)
    dist = tdib.MixtureSameFamily(cat, tdib.MultivariateNormal(mu, torch.diag_embed(std)))
    return dist

class BinaryHead(nn.Module):
  def __init__(self, C):
    super().__init__()
    self.C = C
    self.layer = nn.Linear(C.n_embed, 1)

  def forward(self, x, past_o=None):
    x = self.layer(x)
    return tdib.Bernoulli(logits=x)