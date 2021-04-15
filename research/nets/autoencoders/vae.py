import numpy as np
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from research import utils
from ._base import Autoencoder

class VAE(Autoencoder):
  def __init__(self, env, G):
    super().__init__(env, G)
    self.G = G
    self.z_size = 128
    self.encoder = Encoder(self.z_size, G)
    self.decoder = Decoder(self.z_size, G)
    self._init()

  def loss(self, batch):
    x = batch['lcd']
    x = x.reshape([-1, 1, self.G.lcd_h, self.G.lcd_w])
    z_post = self.encoder(x)
    decoded = self.decoder(z_post.rsample())
    recon_loss = -decoded['lcd'].log_prob(x).mean((1, 2, 3))
    # kl div constraint
    z_prior = thd.Normal(0, 1)
    kl_loss = thd.kl_divergence(z_post, z_prior).mean(-1)
    # full loss and metrics
    loss = (recon_loss + self.G.beta * kl_loss).mean()
    metrics = {'vae_loss': loss, 'loss/recon_lcd': recon_loss.mean(), 'loss/kl': kl_loss.mean()}
    return loss, metrics

  def encode(self, batch, flatten=None):
    x = batch['lcd'][:,None]
    z_post = self.encoder(x).mean
    return z_post

  def decode(self, z):
    return self.decoder(z)

class Encoder(nn.Module):
  def __init__(self, out_size, G):
    super().__init__()
    nf = G.nfilter
    size = (G.lcd_h * G.lcd_w) // 64
    self.net = nn.Sequential(
        nn.Conv2d(1, nf, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(nf, nf, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(nf, nf, 3, 2, padding=1),
        nn.ReLU(),
        nn.Flatten(-3),
        nn.Linear(size*nf, 2*out_size),
    )

  def get_dist(self, x):
    mu, log_std = x.chunk(2, -1)
    std = F.softplus(log_std) + 1e-4
    return thd.Normal(mu, std)

  def forward(self, x):
    return self.get_dist(self.net(x))

class Decoder(nn.Module):
  def __init__(self, in_size, G):
    super().__init__()
    assert G.lcd_h == 16, G.lcd_w == 32
    nf = G.nfilter
    self.net = nn.Sequential(
        nn.ConvTranspose2d(in_size, nf, (2,4), 2),
        nn.ReLU(),
        nn.ConvTranspose2d(nf, nf, 4, 4, padding=0),
        nn.ReLU(),
        nn.Conv2d(nf, nf, 3, 1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(nf, 1, 4, 2, padding=1),
    )

  def forward(self, x):
    x = self.net(x[..., None, None])
    return {'lcd': thd.Bernoulli(logits=x)}