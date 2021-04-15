import inspect
from os import pipe, stat
from re import I
import numpy as np
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from research import utils
from research.nets.common import ResBlock
from torch.optim import Adam
from ._base import Autoencoder

class ArbiterAE(Autoencoder):
  def __init__(self, env, G):
    super().__init__(env, G)
    self.z_size = 128
    state_n = env.observation_space.spaces['pstate'].shape[0]
    self.encoder = Encoder(state_n, self.z_size, G)
    self.decoder = Decoder(state_n, self.z_size, G)
    self._init()

  def unprompted_eval(self, writer=None, itr=None, arbiter=None):
    return {}

  def save(self, dir, batch):
    print("SAVED MODEL", dir)
    path = dir / f'{self.name}.pt'
    jit_enc = th.jit.trace(self.encoder, self._flat_batch(batch))
    th.jit.save(jit_enc, str(path))
    print(path)

  def loss(self, batch):
    z = self.encoder(batch)
    decoded = self.decoder(z)
    recon_losses = {}
    recon_losses['loss/recon_pstate'] = -decoded['pstate'].log_prob(batch['pstate']).mean()
    recon_losses['loss/recon_lcd'] = -decoded['lcd'].log_prob(batch['lcd'][:, None]).mean()
    recon_loss = sum(recon_losses.values())
    metrics = {'loss/recon_total': recon_loss, **recon_losses}
    return recon_loss, metrics

  def encode(self, batch, flatten=None):
    return self.encoder(batch)

  def decode(self, z):
    return self.decoder(z)

class Encoder(nn.Module):
  def __init__(self, state_n, out_size, G):
    super().__init__()
    n = G.hidden_size
    self.state_embed = nn.Sequential(
        nn.Linear(state_n, n),
        nn.ReLU(),
        nn.Linear(n, n),
        nn.ReLU(),
        nn.Linear(n, n),
    )
    nf = G.nfilter
    size = (G.lcd_h * G.lcd_w) // 64
    self.seq = nn.ModuleList([
        nn.Conv2d(1, nf, 3, 2, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Conv2d(nf, nf, 3, 2, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Conv2d(nf, nf, 3, 2, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Flatten(-3),
        nn.Linear(size * nf, out_size),

    ])

  def forward(self, batch):
    state = batch['pstate']
    lcd = batch['lcd']
    emb = self.state_embed(state)
    x = lcd[:, None]
    for layer in self.seq:
      if isinstance(layer, ResBlock):
        x = layer(x, emb)
      else:
        x = layer(x)
    return x

class Decoder(nn.Module):
  def __init__(self, state_n, in_size, G):
    super().__init__()
    assert G.lcd_h == 16, G.lcd_w == 32
    nf = G.nfilter
    self.net = nn.Sequential(
        nn.ConvTranspose2d(in_size, nf, (2, 4), 2),
        nn.ReLU(),
        nn.ConvTranspose2d(nf, nf, 4, 4, padding=0),
        nn.ReLU(),
        nn.Conv2d(nf, nf, 3, 1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(nf, 1, 4, 2, padding=1),
    )
    n = G.hidden_size
    self.state_net = nn.Sequential(
        nn.Linear(in_size, n),
        nn.ReLU(),
        nn.Linear(n, n),
        nn.ReLU(),
        nn.Linear(n, state_n),
    )
    nf = G.nfilter

  def forward(self, x):
    lcd_dist = thd.Bernoulli(logits=self.net(x[..., None, None]))
    state_dist = thd.Normal(self.state_net(x), 1)
    return {'lcd': lcd_dist, 'pstate': state_dist}
