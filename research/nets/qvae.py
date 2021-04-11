from os import stat
from shutil import ignore_patterns
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
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead, aggregate, MultiHead, ConvEmbed
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vq import VectorQuantizer
import utils

class QVAE(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    # encoder -> VQ -> decoder
    self.encoder = Encoder(env, C)
    self.vq = Quantize(C.hidden_size, C.vqK)
    self.decoder = Decoder(env, C)
    self.optimizer = Adam(self.parameters(), C.lr)
    self.env = env
    self.C = C

  def train_step(self, batch, dry=False):
    if dry:
      return {}
    self.optimizer.zero_grad()
    flatter_batch = {key: val.flatten(0, 1) for key, val in batch.items()}
    loss, metrics = self.loss(flatter_batch)
    if not dry:
      loss.backward()
      self.optimizer.step()
    return metrics

  def save(self, dir):
    print("SAVED MODEL", dir)
    path = dir / 'qvae.pt'
    sd = self.state_dict()
    sd['C'] = self.C
    th.save(sd, path)
    print(path)

  def load(self, dir):
    path = dir / 'qvae.pt'
    sd = th.load(path)
    C = sd.pop('C')
    self.load_state_dict(sd)
    print(f'LOADED {path}')

  def evaluate(self, writer, batch, epoch):
    bs = batch['pstate'].shape[0]
    #ebatch = self.flatbatch(batch)
    flatter_batch = {key: val[:8, 0] for key, val in batch.items()}
    _, decoded, _ = self.forward(flatter_batch)
    pred_lcd = 1.0 * (decoded['lcd'].probs > 0.5)[:8]
    lcd = flatter_batch['lcd'][:8,None]
    error = (pred_lcd - lcd + 1.0) / 2.0
    stack = th.cat([lcd, pred_lcd, error], -2)
    writer.add_image('image/decode', utils.combine_imgs(stack, 1, 8)[None], epoch)

    pred_state = decoded['pstate'].mean.detach().cpu()
    true_state = flatter_batch['pstate'].cpu()
    preds = []
    for s in pred_state:
      preds += [self.env.reset(pstate=s)['lcd']]
    truths = []
    for s in true_state:
      truths += [self.env.reset(pstate=s)['lcd']]
    preds = 1.0 * np.stack(preds)
    truths = 1.0 * np.stack(truths)
    error = (preds - truths + 1.0) / 2.0
    stack = np.concatenate([truths, preds, error], -2)[:, None]
    writer.add_image('pstate/decode', utils.combine_imgs(stack, 1, 8)[None], epoch)


  def loss(self, batch, eval=False, return_idxs=False):
    embed_loss, decoded, idxs = self.forward(batch)
    recon_losses = {}
    recon_losses['recon_pstate'] = -decoded['pstate'].log_prob(batch['pstate']).mean()
    recon_losses['recon_lcd'] = -decoded['lcd'].log_prob(batch['lcd'][:,None]).mean()
    recon_loss = sum(recon_losses.values())
    loss = recon_loss + 0.0*embed_loss
    metrics = {'vq_vae_loss': loss, 'embed_loss': embed_loss, **recon_losses, 'recon_loss': recon_loss}
    if eval: metrics['decoded'] = decoded
    if return_idxs: metrics['idxs'] = idxs
    return loss, metrics

  def forward(self, x):
    z_e = self.encoder(x)
    z_q, embed_loss, idxs = self.vq(z_e)
    import ipdb; ipdb.set_trace()
    decoded = self.decoder(z_q)
    return embed_loss, decoded, idxs

class Encoder(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    state_n = env.observation_space.spaces['pstate'].shape[0]
    act_n = env.action_space.shape[0]

    self.state_embed = nn.Sequential(
      nn.Linear(state_n, H),
      nn.ReLU(),
      nn.Linear(H, H),
      nn.ReLU(),
      nn.Linear(H, H),
    )
    self.seq = nn.ModuleList([
        nn.Conv2d(1, H, 3, 1, padding=1),
        ResBlock(H, emb_channels=H),
        nn.Conv2d(H, H, 3, 2, padding=1),
        ResBlock(H, emb_channels=H),
        nn.Conv2d(H, H, 3, 2, padding=1),
        ResBlock(H, emb_channels=H),
        nn.Conv2d(H, C.vqD, 1, 1),
    ])

  def forward(self, batch):
    state = batch['pstate']
    lcd = batch['lcd']
    emb = self.state_embed(state)
    x = lcd[:,None]
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
  def forward(self, x, emb=None):
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = self.conv(x)
    return x

class Decoder(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    state_n = env.observation_space.spaces['pstate'].shape[0]
    self.state_net = nn.Sequential(
      nn.Flatten(-3),
      nn.Linear(C.vqD*4*8, H),
      nn.ReLU(),
      nn.Linear(H, H),
      nn.ReLU(),
      nn.Linear(H, state_n),
    )
    self.net = nn.Sequential(
        Upsample(C.vqD, H),
        nn.ReLU(),
        Upsample(H, H),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, 1, 3, padding=1),
    )
  def forward(self, x):
    lcd_dist = thd.Bernoulli(logits=self.net(x))
    state_dist = thd.Normal(self.state_net(x), 1)
    return {'lcd': lcd_dist, 'pstate': state_dist}

class ResBlock(nn.Module):
  def __init__(self, channels, emb_channels, out_channels=None, dropout=0.0):
    super().__init__()
    self.out_channels = out_channels or channels

    self.in_layers = nn.Sequential(
        nn.GroupNorm(32, channels),
        nn.SiLU(),
        nn.Conv2d(channels, self.out_channels, 3, padding=1)
    )
    self.emb_layers = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_channels, self.out_channels)
    )
    self.out_layers = nn.Sequential(
        nn.GroupNorm(32, self.out_channels),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        utils.zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
    )
    if self.out_channels == channels:
      self.skip_connection = nn.Identity()
    else:
      self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)  # step down size

  def forward(self, x, emb):
    h = self.in_layers(x)
    emb_out = self.emb_layers(emb)[..., None, None]
    h = h + emb_out
    h = self.out_layers(h)
    return self.skip_connection(x) + h

class Quantize(nn.Module):
  """there is no god"""
  def __init__(self, num_hiddens, n_embed):
    super().__init__()
    self.n_embed = n_embed
    self.kld_scale = 5e-4
    self.proj = nn.Linear(num_hiddens, n_embed)

  def forward(self, z):
    #logits = self.proj(z)
    dist = thd.Bernoulli(logits=z)
    z_q = dist.sample()
    z_q += dist.probs - dist.probs.detach()
    # + kl divergence to the prior loss (entropy bonus)
    diff = self.kld_scale * dist.entropy().mean()
    return z_q, diff, z_q
