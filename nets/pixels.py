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
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead, aggregate

class AutoWorldModel(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.encoder = EncoderNet(env, C)
    self.temporal = Transformer(size=self.C.n_embed//2, block_size=self.C.ep_len, dist='mdn', cond=False, C=C)
    self.pixel = Transformer(size=1, block_size=self.C.lcd_h*self.C.lcd_w, dist='binary', cond=True, C=C)
  
  def loss(self, batch):
    BS, EPL, H, W = batch['lcd'].shape
    codes = self.encoder(batch)
    temporal_loss, temporal_dist = self.temporal.loss(codes)
    pix_img = batch['lcd'].view(BS*EPL, H*W, 1)
    timecond = codes.view(BS*EPL, self.C.n_embed//2)
    pixel_loss, pixel_dist = self.pixel.loss(pix_img, timecond)
    return temporal_loss + pixel_loss, (temporal_dist, pixel_dist)

  def sample(self, n):
    codes, code_logp = self.temporal.sample(n)
    BS, EPL, X = codes.shape
    timecond = codes.view(BS*EPL, self.C.n_embed//2)
    pixels, pixel_logp = self.pixel.sample(n, cond=timecond)
    return pixels.view(BS, EPL, 1, self.C.lcd_h, self.C.lcd_w), pixel_logp + code_logp

class EncoderNet(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    state_n = env.observation_space['state'].shape[0]
    act_n = env.action_space.shape[0]
    self.c1 = nn.Conv2d(1, self.C.n_embed//2, kernel_size=3, stride=2, padding=1)
    self.c2 = nn.Conv2d(self.C.n_embed//2, self.C.n_embed//2, kernel_size=3, stride=2, padding=1)
    self.state_in = nn.Linear(state_n, self.C.n_embed//2)
    self.act_in = nn.Linear(act_n, self.C.n_embed//2)
    self.mhdpa = nn.MultiheadAttention(self.C.n_embed//2, 8)
    self.latent_out = nn.Linear(self.C.n_embed//2, self.C.n_embed//8)
    self.ln_f = nn.LayerNorm(C.n_embed//2)
    self.to(C.device)

  def forward(self, batch):
    """expects (BS, EP_LEN, *SHAPE)"""
    BS, EPL = batch['lcd'].shape[:2]
    img = batch['lcd'].view(-1, 1, self.C.lcd_h, self.C.lcd_w)
    img = self.c1(img)
    img = F.relu(img)
    img = self.c2(img).flatten(-2)
    zs = self.state_in(batch['state']).view(-1, self.C.n_embed//2, 1)
    za = self.act_in(batch['acts']).view(-1, self.C.n_embed//2, 1)
    x = torch.cat([img, zs, za], axis=-1)
    x = x.permute(2, 0, 1)  # T, BS, E
    # flatten and cat
    x = self.mhdpa(x, x, x)[0].permute(1, 0, 2) # BS, T, E
    x = self.latent_out(x)
    agg = aggregate(x, dim=1, catdim=-1).view(BS, EPL, self.C.n_embed//2)
    return self.ln_f(agg)

class Transformer(nn.Module):
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

  def loss(self, x, cond=None):
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