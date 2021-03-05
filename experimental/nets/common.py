from define_config import env_fn
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
import torch.distributions.transforms as tran
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
  def __init__(self, in_n, out_n, C):
    super().__init__()
    self.C = C
    self.out_n = out_n
    shape = self.C.mdn_k + 2 * self.out_n * self.C.mdn_k
    self.layer = nn.Linear(in_n, shape)

  def forward(self, x, past_o=None):
    dx = self.C.mdn_k * self.out_n
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
  def __init__(self, in_n, out_n, C):
    super().__init__()
    self.C = C
    self.layer = nn.Linear(in_n, out_n)

  def forward(self, x, past_o=None):
    x = self.layer(x)
    return tdib.Bernoulli(logits=x)

class ConvBinHead(nn.Module):
  def __init__(self, in_n, out_n, C):
    super().__init__()
    self.C = C
    self.in_n = in_n
    self.out_n = out_n
    self.shape = int(self.C.wh_ratio)
    #self.d1 = nn.ConvTranspose2d(self.in_n//self.shape, 64, 7, stride=2)
    #self.d2 = nn.ConvTranspose2d(64, 1, 4, stride=2)
    self.d1 = nn.ConvTranspose2d(self.in_n//self.shape, 64, 4, stride=4)
    self.d2 = nn.ConvTranspose2d(64, 1, 4, stride=4)
    # TODO: try a version like I have below. where it is just in place. perhaps that could smooth things out

  def forward(self, x, past_o=None):
    BS, LEN, C = x.shape
    x = x.reshape(BS * LEN, -1, self.shape, 1)
    x = self.d1(x)
    x = F.relu(x)
    x = self.d2(x)
    x = x.reshape(BS, LEN, -1)
    return tdib.Bernoulli(logits=x)

class ConvEmbed(nn.Module):
  def __init__(self, in_n, out_n, C):
    super().__init__()
    self.C = C
    self.c1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
    self.c2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

  def forward(self, x, past_o=None):
    BS, LEN, C = x.shape
    x = x.reshape(BS * LEN, -1, self.C.lcd_h, self.C.lcd_w)
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
  def __init__(self, in_n, out_n, split, C):
    super().__init__()
    self.C = C
    self.in_n = in_n
    self.out_n = out_n
    self.split = split
    self.layer = nn.Linear(in_n, in_n * 2)
    if self.C.conv_io:
      self.binary = ConvBinHead(in_n, self.split, C)
    else:
      self.binary = BinaryHead(in_n, self.split, C)
    #self.state = FlowHead(in_n, out_n - self.split, C)
    #self.state = GaussHead(in_n, out_n-self.split, C) 
    self.state = MDNHead(in_n, out_n-self.split, C) 
    #self.state = GPT(1, block_size=out_n-self.split, dist='mdn', cond=in_n, C=C) 

  def forward(self, x, past_o=None):
    xb, xs = self.layer(x).chunk(2, -1)
    bin = self.binary(xb) if 'image' in self.C.subset else None
    state = self.state(xs) if 'state' in self.C.subset else None
    #return bin, GPTDist(self.state, xs)
    return bin, state