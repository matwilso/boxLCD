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

  def __init__(self, C):
    super().__init__()
    assert C.n_embed % C.n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(C.n_embed, C.n_embed)
    self.query = nn.Linear(C.n_embed, C.n_embed)
    self.value = nn.Linear(C.n_embed, C.n_embed)
    # output projection
    self.proj = nn.Linear(C.n_embed, C.n_embed)
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", torch.tril(torch.ones(C.block_size, C.block_size)).view(1, 1, C.block_size, C.block_size))
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

  def __init__(self, C):
    super().__init__()
    self.ln1 = nn.LayerNorm(C.n_embed)
    self.ln2 = nn.LayerNorm(C.n_embed)
    self.attn = CausalSelfAttention(C)
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
  def __init__(self, obs_n, C):
    super().__init__()
    self.C = C
    self.obs_n = obs_n
    self.layer = nn.Linear(C.n_embed, 2 * obs_n)

  def forward(self, x, past_o=None):
    mu, log_std = self.layer(x).chunk(2, -1)
    std = F.softplus(log_std) + self.C.min_std
    if past_o is not None:
      mu = mu + past_o
    #dist = tdib.Independent(tdib.Normal(mu, std), 1)
    #dist = tdib.Normal(mu, std)
    dist = tdib.MultivariateNormal(mu, torch.diag_embed(std))
    return dist

class MDNHead(nn.Module):
  def __init__(self, obs_n, C):
    super().__init__()
    self.C = C
    self.obs_n = obs_n
    shape = self.C.mdn_k + 2*self.obs_n*self.C.mdn_k
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
      mu = mu + past_o[...,None,:]
    cat = tdib.Categorical(logits=logits)
    dist = tdib.MixtureSameFamily(cat, tdib.MultivariateNormal(mu, torch.diag_embed(std)))
    return dist

class Transformer(nn.Module):
  """  the full GPT language model, with a context size of block_size """

  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.obs_n = env.observation_space.shape[0]
    self.shape = self.obs_n + env.action_space.shape[0] + 1
    # input embedding stem
    self.embed = nn.Linear(self.shape, C.n_embed, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[Block(C) for _ in range(C.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(C.n_embed)
    if self.C.mdn_k == 1:
      self.dist_head = GaussHead(self.obs_n, C)
    else:
      self.dist_head = MDNHead(self.obs_n, C)
    #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    self.to(C.device)

  def append_location(self, x):
    """add xy coords to every pixel"""
    X = torch.linspace(-1, 1, x.shape[-2])
    return torch.cat([x, X[None, ..., None].repeat_interleave(x.shape[0], 0).to(x.device)], -1)

  def forward(self, inp):
    x = inp
    x = self.append_location(x)
    # forward the GPT model
    x = self.embed(x)
    # add padding on left so that we can't see ourself.
    x = self.blocks(x)
    logits = self.ln_f(x)
    # TODO: probably return logits as well.
    return self.dist_head(logits, past_o=inp[...,:self.obs_n] if self.C.dist_delta else None)

  def nll(self, batch):
    # TODO: clean this shifting to happen in model probably
    o, a = batch['o'], batch['a']
    batch_size = o.shape[0]
    x = torch.cat([o, a], -1)
    shifted = torch.cat([torch.zeros(batch_size, 1, x.shape[-1]).to(self.C.device), x[:, :-1]], dim=1)
    dist = self.forward(shifted)
    return -dist.log_prob(o).mean(), dist

  def sample(self, n, prompts=None):
    # TODO: feed act_n
    with torch.no_grad():
      samples = torch.zeros(n, self.C.ep_len, self.obs_n).to(self.C.device)
      acts = (torch.rand(samples.shape[:-1]) * 2 - 1).to(self.C.device)[..., None]

      start = 0
      if prompts is not None:
        n, k, _ = prompts.shape
        samples[:n, 1:k+1, :] = torch.as_tensor(prompts, dtype=torch.float32).to(samples.device)
        start = k

      for i in range(start, self.C.ep_len-1):
        x = torch.cat([samples, acts], -1)
        dist = self.forward(x)
        if self.C.sample_sample:
          samples[:, i + 1] = dist.sample()[:,i]
        else:
          samples[:, i + 1] = dist.mean[:,i]
        if i == self.C.ep_len-2:
          logp = dist.log_prob(samples)

    return samples.cpu(), logp.mean().item()