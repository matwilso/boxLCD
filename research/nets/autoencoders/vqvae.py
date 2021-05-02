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
from research.nets.common import GaussHead, MDNHead, CausalSelfAttention, TransformerBlock, BinaryHead, aggregate, MultiHead, ConvEmbed
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from research.nets.quantize import VectorQuantizer
from research import utils
from research.nets.common import ResBlock

# THIS IS NOT USED AT THE MOMENT. based on https://github.com/matwilso/generative_models#vq-vae

class VQVAE(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    H = G.hidden_size
    # encoder -> VQ -> decoder
    self.encoder = Encoder(env, G)
    self.vq = VectorQuantizer(G.vqK, G.vqD, G.beta, G)
    self.decoder = Decoder(env, G)
    self.optimizer = Adam(self.parameters(), G.lr)
    self.env = env
    self.G = G

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

  def evaluate(self, writer, batch, epoch):
    bs = batch['proprio'].shape[0]
    #ebatch = self.flatbatch(batch)
    flatter_batch = {key: val[:8, 0] for key, val in batch.items()}
    _, decoded, _, idxs = self.forward(flatter_batch)
    pred_lcd = 1.0 * (decoded['lcd'].probs > 0.5)[:8]
    lcd = flatter_batch['lcd'][:8,None]
    error = (pred_lcd - lcd + 1.0) / 2.0
    stack = th.cat([lcd, pred_lcd, error], -2)
    writer.add_image('image/decode', utils.combine_imgs(stack, 1, 8)[None], epoch)

    import ipdb; ipdb.set_trace()
    pred_state = decoded['proprio'].mean[:8,0].detach().cpu()
    true_state = flatter_batch['proprio'][:8,0].cpu()
    preds = []
    for s in pred_state:
      preds += [self.env.reset(proprio=s)['lcd']]
    truths = []
    for s in true_state:
      truths += [self.env.reset(proprio=s)['lcd']]
    preds = 1.0 * np.stack(preds)
    truths = 1.0 * np.stack(truths)
    error = (preds - truths + 1.0) / 2.0
    stack = np.concatenate([truths, preds, error], -2)[:, None]
    writer.add_image('proprio/decode', utils.combine_imgs(stack, 1, 8)[None], epoch)


  def loss(self, batch, eval=False, return_idxs=False):
    embed_loss, decoded, perplexity, idxs = self.forward(batch)
    recon_losses = {}
    recon_losses['recon_proprio'] = -decoded['proprio'].log_prob(batch['proprio']).mean()
    recon_losses['recon_lcd'] = -decoded['lcd'].log_prob(batch['lcd'][:,None]).mean()
    recon_loss = sum(recon_losses.values())
    loss = recon_loss + embed_loss
    metrics = {'vq_vae_loss': loss, 'embed_loss': embed_loss, 'perplexity': perplexity, **recon_losses, 'recon_loss': recon_loss}
    if eval: metrics['decoded'] = decoded
    if return_idxs: metrics['idxs'] = idxs
    return loss, metrics

  def forward(self, x):
    z_e = self.encoder(x)
    embed_loss, z_q, perplexity, idxs = self.vq(z_e)
    decoded = self.decoder(z_q)
    return embed_loss, decoded, perplexity, idxs

class Encoder(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    H = G.hidden_size
    state_n = env.observation_space.spaces['proprio'].shape[0]
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
    ])

  def forward(self, batch):
    state = batch['proprio']
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
  def __init__(self, env, G):
    super().__init__()
    H = G.hidden_size
    state_n = env.observation_space.spaces['proprio'].shape[0]
    self.state_net = nn.Sequential(
      nn.Flatten(-3),
      nn.Linear(G.vqD*4*8, H),
      nn.ReLU(),
      nn.Linear(H, H),
      nn.ReLU(),
      nn.Linear(H, state_n),
    )
    self.net = nn.Sequential(
        Upsample(G.vqD, H),
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
    return {'lcd': lcd_dist, 'proprio': state_dist}
