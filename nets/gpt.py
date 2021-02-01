import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead

class GPT(nn.Module):
  def __init__(self, size, block_size, dist, cond=False, C=None):
    super().__init__()
    self.C = C
    self.size = size
    self.block_size = block_size
    self.cond = cond
    self.dist = dist
    # input embedding stem
    self.embed = nn.Linear(size+1, C.n_embed//2 if cond else C.n_embed, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[Block(block_size, C) for _ in range(C.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(C.n_embed)
    if dist == 'gauss':
      self.dist_head = GaussHead(self.size, C)
    elif dist == 'mdn':
      self.dist_head = MDNHead(self.size, C)
    elif dist == 'binary':
      self.dist_head = BinaryHead(C)
    self.to(C.device)

  def append_location(self, x):
    """add loc coords to every elem"""
    X = torch.linspace(-1, 1, x.shape[-2])
    return torch.cat([x, X[None, ..., None].repeat_interleave(x.shape[0], 0).to(x.device)], -1)

  def forward(self, x, cond=None):
    x = self.append_location(x)
    # forward the GPT model
    x = self.embed(x)
    if self.cond:
      x = torch.cat([x, cond[:,None].repeat_interleave(self.block_size, 1)], -1)
    # add padding on left so that we can't see ourself.
    x = self.blocks(x)
    logits = self.ln_f(x)
    # TODO: probably return logits as well.
    return self.dist_head(logits, past_o=inp[...,:self.size] if self.C.dist_delta else None)

  def nll(self, x, cond=None):
    BS, EPL, E = x.shape
    shifted = torch.cat([torch.zeros(BS, 1, E).to(self.C.device), x[:, :-1]], dim=1)
    dist = self.forward(shifted, cond)
    return -dist.log_prob(x).mean(), dist

  def sample(self, n, cond=None):
    # TODO: feed act_n
    with torch.no_grad():
      if cond is not None:
        n = cond.shape[0]
      samples = torch.zeros(n, self.block_size, self.size).to(self.C.device)
      #acts = (torch.rand(samples.shape[:-1]) * 2 - 1).to(self.C.device)[..., None]
      for i in range(self.block_size-1):
        dist = self.forward(samples, cond)
        if self.C.sample_sample or self.dist == 'binary':
          samples[:, i + 1] = dist.sample()[:,i]
        else:
          samples[:, i + 1] = dist.mean[:,i]
        if i == self.block_size-2:
          logp = dist.log_prob(samples)

    return samples, logp.mean().item()