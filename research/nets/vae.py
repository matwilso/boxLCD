import sys
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import torch as torchvision
from torch.optim import Adam
from itertools import chain, count
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
import utils

class VAE(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.z_size = 128
    self.encoder = Encoder(self.z_size, C)
    self.decoder = Decoder(self.z_size, C)
    self.optimizer = Adam(self.parameters(), lr=C.lr)

  def save(self, dir):
    print("SAVED MODEL", dir)
    path = dir / 'vae.pt'
    th.save(self.state_dict(), path)
    print(path)

  def load(self, path):
    path = path / 'vae.pt'
    self.load_state_dict(th.load(path))
    print(f'LOADED {path}')

  def loss(self, batch):
    x = batch['lcd']
    x = x.reshape([-1, 1, self.C.lcd_h, self.C.lcd_w])
    z_post = self.encoder(x)
    decoded = self.decoder(z_post.rsample())
    recon_loss = -thd.Bernoulli(logits=decoded).log_prob(x).mean((1, 2, 3))
    # kl div constraint
    z_prior = thd.Normal(0, 1)
    kl_loss = thd.kl_divergence(z_post, z_prior).mean(-1)
    # full loss and metrics
    loss = (recon_loss + self.C.beta * kl_loss).mean()
    metrics = {'vae_loss': loss, 'recon_loss': recon_loss.mean(), 'kl_loss': kl_loss.mean()}
    return loss, metrics

  def train_step(self, batch, dry=False):
    self.optimizer.zero_grad()
    loss, metrics = self.loss(batch)
    if not dry:
      loss.backward()
      self.optimizer.step()
    return metrics

  def evaluate(self, writer, batch, epoch):
    """run samples and other evaluations"""
    x = batch['lcd'].reshape([-1, 1, self.C.lcd_h, self.C.lcd_w])
    z_post = self.encoder(x[:8])
    recon = self._decode(z_post.mean)
    recon = th.cat([x[:8].cpu(), recon], 0)
    writer.add_image('reconstruction', utils.combine_imgs(recon, 2, 8)[None], epoch)

  def _decode(self, x):
    return 1.0 * (self.decoder(x).exp() > 0.5).cpu()

class Encoder(nn.Module):
  def __init__(self, out_size, C):
    super().__init__()
    H = C.nfilter
    size = (C.lcd_h * C.lcd_w) // 64
    self.net = nn.Sequential(
        nn.Conv2d(1, H, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 2, padding=1),
        nn.ReLU(),
        nn.Flatten(-3),
        nn.Linear(size*H, 2*out_size),
    )

  def get_dist(self, x):
    mu, log_std = x.chunk(2, -1)
    std = F.softplus(log_std) + 1e-4
    return thd.Normal(mu, std)

  def forward(self, x):
    return self.get_dist(self.net(x))

class Decoder(nn.Module):
  def __init__(self, in_size, C):
    super().__init__()
    H = C.nfilter
    assert C.lcd_h == 16, C.lcd_w == 32
    self.net = nn.Sequential(
        nn.ConvTranspose2d(in_size, H, (2,4), 2),
        nn.ReLU(),
        nn.ConvTranspose2d(H, H, 4, 4, padding=0),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(H, 1, 4, 2, padding=1),
    )

  def forward(self, x):
    x = self.net(x[..., None, None])
    return x
