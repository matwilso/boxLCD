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

# TODO: VQ VAE may be worth doing. but maybe as a separate repo.
from gms.nets import E1, D1
from gms import utils


# TODO: record bits/dim
# TODO: try interpolation
# TODO: barebon, no residual block version


class CausalSelfAttention(nn.Module):
  """
  A vanilla multi-head masked self-attention layer with a projection at the end.
  It is possible to use torch.nn.MultiheadAttention here but I am including an
  explicit implementation here to show that there is nothing too scary here.
  """

  def __init__(self, F):
    super().__init__()
    assert F.n_embed % F.n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(F.n_embed, F.n_embed)
    self.query = nn.Linear(F.n_embed, F.n_embed)
    self.value = nn.Linear(F.n_embed, F.n_embed)
    # output projection
    self.proj = nn.Linear(F.n_embed, F.n_embed)
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", torch.tril(torch.ones(F.block_size, F.block_size)).view(1, 1, F.block_size, F.block_size))
    self.F = F

  def forward(self, x, layer_past=None):
    B, T, C = x.size()
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    k = self.key(x).view(B, T, self.F.n_head, C // self.F.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.F.n_head, C // self.F.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.F.n_head, C // self.F.n_head).transpose(1, 2)  # (B, nh, T, hs)
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

  def __init__(self, F):
    super().__init__()
    self.ln1 = nn.LayerNorm(F.n_embed)
    self.ln2 = nn.LayerNorm(F.n_embed)
    self.attn = CausalSelfAttention(F)
    self.mlp = nn.Sequential(
        nn.Linear(F.n_embed, 4 * F.n_embed),
        nn.GELU(),
        nn.Linear(4 * F.n_embed, F.n_embed),
    )

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x

class Transformer(nn.Module):
  """  the full GPT language model, with a context size of block_size """

  def __init__(self, env, F):
    super().__init__()
    self.obs_n = env.observation_space.shape[0]
    self.shape = self.obs_n + env.action_space.shape[0] + 1
    # input embedding stem
    self.embed = nn.Linear(self.shape, F.n_embed, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[Block(F) for _ in range(F.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(F.n_embed)
    self.head = nn.Linear(F.n_embed, 2 * self.obs_n, bias=False)
    #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    self.F = F
    self.to(F.device)

  def append_location(self, x):
    """add xy coords to every pixel"""
    X = torch.linspace(-1, 1, x.shape[-2])
    return torch.cat([x, X[None, ..., None].repeat_interleave(x.shape[0], 0).to(x.device)], -1)

  def forward(self, x):
    x = self.append_location(x)
    # forward the GPT model
    x = self.embed(x)
    # add padding on left so that we can't see ourself.
    x = self.blocks(x)
    x = self.ln_f(x)
    x = self.head(x)
    return x

  def get_dist(self, mu, log_std, past_o):
    std = F.softplus(log_std) + self.F.min_std
    mu = mu + past_o[..., :self.obs_n]
    dist = tdib.Normal(mu, std)
    return dist

  def nll(self, batch):
    # TODO: clean this shifting to happen in model probably
    o, a = batch['o'], batch['a']
    batch_size = o.shape[0]
    x = torch.cat([o, a], -1)
    shifted = torch.cat([torch.zeros(batch_size, 1, x.shape[-1]).to(self.F.device), x[:, :-1]], dim=1)
    mu, log_std = self.forward(shifted).chunk(2, -1)
    dist = self.get_dist(mu, log_std, shifted)
    return -dist.log_prob(o).mean()

  def sample(self, n, prompts=None):
    # TODO: feed act_n
    with torch.no_grad():
      samples = torch.zeros(n, 200, self.obs_n).to(self.F.device)
      acts = (torch.rand(samples.shape[:-1]) * 2 - 1).to(self.F.device)[..., None]

      start = 0
      if prompts is not None:
        n, k, _ = prompts.shape
        samples[:n, 1:k+1, :] = torch.as_tensor(prompts, dtype=torch.float32).to(samples.device)
        start = k

      for i in range(start, 199):
        x = torch.cat([samples, acts], -1)
        mu, log_std = self.forward(x).chunk(2, -1)
        dist = self.get_dist(mu[:, i+1], log_std[:, i+1], past_o=samples[:, i])
        samples[:, i + 1] = dist.sample()
        if i == 198:
          logp = self.get_dist(mu, log_std, past_o=samples)

    return samples.cpu(), logp.mean.mean().item()