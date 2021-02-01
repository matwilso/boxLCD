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

from nets.common import GaussHead, MDNHead

class StateTransformer(nn.Module):
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