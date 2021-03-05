import sys
from collections import defaultdict
import numpy as np
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

class EncoderNet(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    state_n = env.observation_space['state'].shape[0]
    act_n = env.action_space.shape[0]
    self.c1 = nn.Conv2d(1, self.C.n_embed // 2, kernel_size=3, stride=2, padding=1)
    self.c2 = nn.Conv2d(self.C.n_embed // 2, self.C.n_embed // 2, kernel_size=3, stride=2, padding=1)
    self.state_in = nn.Linear(state_n, self.C.n_embed // 2)
    self.act_in = nn.Linear(act_n, self.C.n_embed // 2)
    self.mhdpa = nn.MultiheadAttention(self.C.n_embed // 2, 8)
    self.latent_out = nn.Linear(self.C.n_embed // 2, self.C.n_embed // 4)
    #self.ln_f = nn.LayerNorm(C.n_embed // 2)
    self.head = GaussHead(self.C.n_embed, self.C.n_embed//2, self.C)
    #self.post_agg = nn.Linear(self.C.n_embed // 2, 2*self.C.n_embed)
    self.to(C.device)

  def forward(self, batch):
    """expects (BS, H, W)"""
    BS = batch['lcd'].shape[0]
    img = batch['lcd'].view(-1, 1, self.C.lcd_h, self.C.lcd_w)
    img = self.c1(img)
    img = F.relu(img)
    img = self.c2(img).flatten(-2)
    zs = self.state_in(batch['state']).view(-1, self.C.n_embed // 2, 1)
    za = self.act_in(batch['acts']).view(-1, self.C.n_embed // 2, 1)
    x = torch.cat([img, zs, za], axis=-1)
    x = x.permute(2, 0, 1)  # T, BS, E
    # flatten and cat
    x = self.mhdpa(x, x, x)[0].permute(1, 0, 2)  # BS, T, E
    x = self.latent_out(x)
    agg = aggregate(x, dim=1, catdim=-1).view(BS, self.C.n_embed)
    return self.head(agg)

class PixelVAE(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.encoder = EncoderNet(env, C)
    self.decoder = GPT(size=1, block_size=self.C.lcd_h*self.C.lcd_w, dist='binary', cond=self.C.n_embed//2, C=C)
    self.to(C.device)

  #def forward(self, batch):
  #  x = self.encoder(batch)
  #  import ipdb; ipdb.set_trace()

  def loss(self, batch):
    BS, H, W = batch['lcd'].shape
    enc_dist = self.encoder(batch)
    samples = enc_dist.mean
    #samples = enc_dist.rsample()
    recon_loss, dist = self.decoder.loss(batch['lcd'].view(BS, H*W, 1), cond=samples)
    recon_loss = recon_loss.mean((1,2))
    #prior = tdib.MultivariateNormal(torch.zeros_like(enc_dist.mean), torch.diag_embed(torch.ones_like(enc_dist.stddev)))
    #prior_loss = tdib.kl_divergence(enc_dist, prior)
    prior_loss = 0.0
    return (recon_loss + prior_loss).mean(), dist

  def sample(self, n):
    pass