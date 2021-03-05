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
from nets.gpt import GPT

class AutoWorld2(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.temporal = GPT(size=self.C.lcd_h*self.C.lcd_w, block_size=self.C.ep_len, dist=self.C.decode, C=C)
    #self.temporal = GPT(size=self.C.lcd_h*self.C.lcd_w, block_size=self.C.ep_len, dist='custom', C=C)
  
  def loss(self, batch):
    BS, EPL, H, W = batch['lcd'].shape
    lcd = batch['lcd'].flatten(-2)
    # should the dist here be independent? prolly not
    temporal_loss, temporal_dist = self.temporal.loss(lcd)
    return temporal_loss.mean(), temporal_dist

  def sample(self, n, prompts=None):
    pixels, logp = self.temporal.sample(n, prompts=prompts)
    return pixels.reshape(n, -1, 1, self.C.lcd_h, self.C.lcd_w), logp

class AutoWorldModel(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.encoder = EncoderNet(env, C)
    self.temporal = GPT(size=self.C.n_embed//2, block_size=self.C.ep_len, dist='mdn', cond=None, C=C)
    self.pixel = GPT(size=1, block_size=self.C.lcd_h*self.C.lcd_w, dist='binary', cond=self.C.n_embed//2, C=C)
  
  def loss(self, batch):
    BS, EPL, H, W = batch['lcd'].shape
    codes = self.encoder(batch)
    temporal_loss, temporal_dist = self.temporal.loss(codes)
    pix_img = batch['lcd'].reshape(BS*EPL, H*W, 1)
    timecond = codes.reshape(BS*EPL, self.C.n_embed//2)
    pixel_loss, pixel_dist = self.pixel.loss(pix_img, timecond)
    return temporal_loss.mean() + pixel_loss.mean(), (temporal_dist, pixel_dist)

  def sample(self, n):
    # TODO: compare sampled codes vs. ones that we actually see during training.
    codes, code_logp = self.temporal.sample(n)
    BS, EPL, X = codes.shape
    timecond = codes.reshape(BS*EPL, self.C.n_embed//2)
    pixels, pixel_logp = self.pixel.sample(n, cond=timecond)
    return pixels.reshape(BS, EPL, 1, self.C.lcd_h, self.C.lcd_w), pixel_logp + code_logp

class EncoderNet(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    state_n = env.observation_space['state'].shape[0]
    act_n = env.action_space.shape[0]
    self.c1 = nn.Conv2d(1, self.C.n_embed//2, kernel_size=3, stride=2, padding=1)
    self.c2 = nn.Conv2d(self.C.n_embed//2, self.C.n_embed//16, kernel_size=3, stride=2, padding=1)
    self.state_in = nn.Linear(state_n, self.C.n_embed//2)
    self.act_in = nn.Linear(act_n, self.C.n_embed//2)
    self.mhdpa = nn.MultiheadAttention(self.C.n_embed//2, 8)
    self.latent_out = nn.Linear(self.C.n_embed, self.C.n_embed//2)
    self.ln_f = nn.LayerNorm(C.n_embed//2)
    self.lout = nn.Linear(self.C.n_embed//2, self.C.n_embed//2, bias=False)
    self.to(C.device)

  def forward(self, batch):
    """expects (BS, EP_LEN, *SHAPE)"""
    BS, EPL = batch['lcd'].shape[:2]
    img = batch['lcd'].reshape(-1, 1, self.C.lcd_h, self.C.lcd_w)
    img = self.c1(img)
    img = F.relu(img)
    img = self.c2(img).flatten(-3)
    zs = self.state_in(batch['state']).reshape(-1, self.C.n_embed//2, 1)
    za = self.act_in(batch['acts']).reshape(-1, self.C.n_embed//2, 1)
    #x = torch.cat([img, zs, za], axis=-1)
    #x = x.permute(2, 0, 1)  # T, BS, E
    # flatten and cat
    #x = self.mhdpa(x, x, x)[0].permute(1, 0, 2) # BS, T, E
    #x = self.latent_out(x)
    #agg = aggregate(x, dim=1, catdim=-1).reshape(BS, EPL, self.C.n_embed//2)
    #agg = x.mean(1).reshape(BS, EPL, self.C.n_embed//2)
    agg = self.latent_out(img).reshape(BS, EPL, self.C.n_embed//2)
    return agg
    #return self.lout(agg)
    #return self.lout(agg)
    #return self.ln_f(agg)
