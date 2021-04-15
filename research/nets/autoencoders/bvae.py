import numpy as np
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from research import utils
from research.nets.common import ResBlock
from .quantize import BinaryQuantize
from ._base import Autoencoder

class BVAE(Autoencoder):
  def __init__(self, env, G):
    super().__init__(env, G)
    # encoder -> binary -> decoder
    self.encoder = Encoder(env, G)
    self.vq = BinaryQuantize(G.hidden_size, G.vqK)
    self.decoder = Decoder(env, G)
    self.z_size = 4 * 8 * G.vqD
    self._init()

  def loss(self, batch):
    # autoencode
    z_e = self.encoder(batch)
    z_q, entropy, probs = self.vq(z_e)
    decoded = self.decoder(z_q)
    # compute losses
    recon_losses = {}
    recon_losses['loss/recon_pstate'] = -decoded['pstate'].log_prob(batch['pstate']).mean()
    recon_losses['loss/recon_lcd'] = -decoded['lcd'].log_prob(batch['lcd'][:, None]).mean()
    recon_loss = sum(recon_losses.values())
    loss = recon_loss - self.G.entropy_bonus*entropy
    metrics = {'loss/total': loss, 'loss/entropy': entropy, **recon_losses, 'loss/recon_total': recon_loss, 'bvae_abs_probs': th.abs(probs-0.5).mean()}
    return loss, metrics

  def encode(self, batch, flatten=True):
    shape = batch['lcd']
    if len(shape) == 4:
      batch = {key: val.clone().flatten(0, 1) for key, val in batch.keys()}
    batch['lcd'].reshape
    z_e = self.encoder(batch)
    # return z_e.flatten(-3)
    z_q, entropy, probs = self.vq(z_e)
    if flatten:
      z_q = z_q.flatten(-3)
      assert z_q.shape[-1] == self.z_size, 'encode shape should equal the z_size. probably forgot to change one.'
    # if len(shape) == 4:
    #  import ipdb; ipdb.set_trace()
    #  return z_q.reshape([*shape[:2], z_q.shape[1:]])
    return z_q

  def decode(self, z_q):
    decoded = self.decoder(z_q)
    return decoded

class Encoder(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    state_n = env.observation_space.spaces['pstate'].shape[0]
    n = G.hidden_size
    self.state_embed = nn.Sequential(
        nn.Linear(state_n, n),
        nn.ReLU(),
        nn.Linear(n, n),
        nn.ReLU(),
        nn.Linear(n, n),
    )
    nf = G.nfilter
    self.seq = nn.ModuleList([
        nn.Conv2d(1, nf, 3, 1, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Conv2d(nf, nf, 3, 2, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Conv2d(nf, nf, 3, 2, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Conv2d(nf, G.vqD, 1, 1),
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

class Upsample(nn.Module):
  """double the size of the input"""
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
  def forward(self, x):
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = self.conv(x)
    return x

class Decoder(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    state_n = env.observation_space.spaces['pstate'].shape[0]
    n = G.hidden_size
    self.state_net = nn.Sequential(
        nn.Flatten(-3),
        nn.Linear(G.vqD * 4 * 8, n),
        nn.ReLU(),
        nn.Linear(n, n),
        nn.ReLU(),
        nn.Linear(n, state_n),
    )
    nf = G.nfilter
    self.net = nn.Sequential(
        Upsample(G.vqD, nf),
        nn.ReLU(),
        Upsample(nf, nf),
        nn.ReLU(),
        nn.Conv2d(nf, nf, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(nf, 1, 3, padding=1),
    )

  def forward(self, x):
    lcd_dist = thd.Bernoulli(logits=self.net(x))
    state_dist = thd.Normal(self.state_net(x), 1)
    return {'lcd': lcd_dist, 'pstate': state_dist}

class StateEncoder(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    H = G.hidden_size
    state_n = env.observation_space.spaces['pstate'].shape[0]
    self.state_embed = nn.Sequential(
        nn.Linear(state_n, H),
        nn.ReLU(),
        nn.Linear(H, H),
        nn.ReLU(),
        nn.Linear(H, H),
    )
  def forward(self, batch):
    state = batch['pstate']
    x = self.state_embed(state)
    return x

class StateDecoder(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    n = G.hidden_size
    state_n = env.observation_space.spaces['pstate'].shape[0]
    self.state_net = nn.Sequential(
        nn.Linear(G.vqK, n),
        nn.ReLU(),
        nn.Linear(n, n),
        nn.ReLU(),
        nn.Linear(n, state_n),
    )
  def forward(self, x):
    return thd.Normal(self.state_net(x), 1)
