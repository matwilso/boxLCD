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
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block

class AutoWorldModel(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.encoder = EncoderNet(env, C)
    self.temporal = TemporalTransformer(64, C)
    self.pixel = PixelTransformer(C)
    self.to(C.device)
  
  def forward(self):
    pass

  def nll(self, batch):
    codes = self.encoder(batch)
    import ipdb; ipdb.set_trace()
    logits, dist = self.temporal(codes)

  def sample(self):
    pass

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

class EncoderNet(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    state_n = env.observation_space['state'].shape[0]
    act_n = env.action_space.shape[0]
    self.c1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
    self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
    self.state_in = nn.Linear(state_n, 64)
    self.act_in = nn.Linear(act_n, 64)
    self.mhdpa = nn.MultiheadAttention(64, 8)
    self.latent_out = nn.Linear(64, 16)

  def forward(self, batch):
    """expects (BS, EP_LEN, *SHAPE)"""
    BS, EP_LEN = batch['lcd'].shape[:2]
    img = batch['lcd'].view(-1, 1, self.C.lcd_h, self.C.lcd_w)
    img = self.c1(img)
    img = F.relu(img)
    img = self.c2(img).flatten(-2)
    zs = self.state_in(batch['state']).view(-1, 64, 1)
    za = self.act_in(batch['acts']).view(-1, 64, 1)
    x = torch.cat([img, zs, za], axis=-1)
    x = x.permute(2, 0, 1)  # T, BS, E
    # flatten and cat
    x = self.mhdpa(x, x, x)[0].permute(1, 0, 2) # BS, T, E
    x = self.latent_out(x)
    agg = aggregate(x, dim=1, catdim=-1)
    return agg.view(BS, EP_LEN, 16*4)

class TemporalTransformer(nn.Module):
  def __init__(self, size, C):
    super().__init__()
    self.C = C
    self.size = size
    # input embedding stem
    self.embed = nn.Linear(size+1, C.n_embed, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[Block(C) for _ in range(C.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(C.n_embed)
    if self.C.mdn_k == 1:
      self.dist_head = GaussHead(self.size, C)
    else:
      self.dist_head = MDNHead(self.size, C)
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
    return logits, self.dist_head(logits, past_o=inp[...,:self.size] if self.C.dist_delta else None)

  def nll(self, batch):
    # TODO: clean this shifting to happen in model probably
    o, a = batch['o'], batch['a']
    batch_size = o.shape[0]
    x = torch.cat([o, a], -1)
    shifted = torch.cat([torch.zeros(batch_size, 1, x.shape[-1]).to(self.C.device), x[:, :-1]], dim=1)
    logits, dist = self.forward(shifted)
    return -dist.log_prob(o).mean(), dist

  def sample(self, n, prompts=None):
    # TODO: feed act_n
    with torch.no_grad():
      samples = torch.zeros(n, self.C.ep_len, self.size).to(self.C.device)
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


class PixelTransformer(nn.Module):
  def __init__(self, C):
    super().__init__()
    # input embedding stem
    self.pixel_emb = nn.Conv2d(3, C.n_embed, kernel_size=1, stride=1)
    # transformer
    self.blocks = nn.Sequential(*[Block(C) for _ in range(C.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(C.n_embed)
    self.head = nn.Conv2d(C.n_embed, 1, kernel_size=1, stride=1, bias=False)
    #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    self.C = C

  def append_location(self, x):
    """add xy coords to every pixel"""
    XY = torch.stack(torch.meshgrid(torch.linspace(0, 1, x.shape[-2]), torch.linspace(0, 1, x.shape[-1])), 0)
    return torch.cat([x, XY[None].repeat_interleave(x.shape[0], 0).to(x.device)], 1)

  def forward(self, x):
    batch_size = x.shape[0]
    x = self.append_location(x)
    # forward the GPT model
    x = self.pixel_emb(x)
    x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, 28*28, -1)
    # add padding on left so that we can't see ourself.
    x = torch.cat([torch.zeros(batch_size, 1, self.C.n_embed).to(self.C.device), x[:, :-1]], dim=1)
    x = self.blocks(x)
    x = self.ln_f(x)
    x = x.permute(0, 2, 1).view(batch_size, -1, 28, 28)
    x = self.head(x)
    return x

  def nll(self, x):
    x = x[0]
    logits = self.forward(x)
    return F.binary_cross_entropy_with_logits(logits, x)

  def sample(self, n):
    imgs = []
    with torch.no_grad():
      samples = torch.zeros(n, 1, 28, 28).to(self.C.device)
      for r in range(28):
        for c in range(28):
          logits = self.forward(samples)[:, :, r, c]
          probs = torch.sigmoid(logits)
          samples[:, :, r, c] = torch.bernoulli(probs)
          imgs += [samples.cpu()]
    imgs = np.stack([img.numpy() for img in imgs], axis=1)
    return samples.cpu(), imgs

