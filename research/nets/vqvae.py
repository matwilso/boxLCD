from re import I
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

class VQVAE(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    # encoder -> VQ -> decoder
    self.encoder = Encoder(C)
    self.vq = VectorQuantizer(C.vqK, C.vqD, C.beta, C)
    self.decoder = Decoder(C)
    self.optimizer = Adam(self.parameters(), C.lr)

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

  def save(self, *args, **kwargs):
    pass

  def loss(self, batch, eval=False, return_idxs=False):
    x = batch['lcd']
    embed_loss, decoded, perplexity, idxs = self.forward(x)
    recon_loss = -thd.Bernoulli(logits=decoded).log_prob(x).mean()
    loss = recon_loss + embed_loss
    prior_loss = th.zeros(1)
    metrics = {'vq_vae_loss': loss, 'recon_loss': recon_loss, 'embed_loss': embed_loss, 'perplexity': perplexity, 'prior_loss': prior_loss}
    if eval:
      metrics['decoded'] = decoded
    if return_idxs:
      metrics['idxs'] = idxs
    return loss, metrics

  def forward(self, x):
    z_e = self.encoder(x)
    embed_loss, z_q, perplexity, idxs = self.vq(z_e)
    decoded = self.decoder(z_q)
    return embed_loss, decoded, perplexity, idxs

  def evaluate(self, writer, batch, epoch):
    flatter_batch = {key: val[:8, 0] for key, val in batch.items()}
    lcd = flatter_batch['lcd']
    _, decoded, _, _ = self.forward(lcd)
    #recon = 1.0 * (decoded.exp() > 0.5).cpu()
    #recon = th.sigmoid(decoded) > 0.5
    recon = 1.0 * (th.sigmoid(decoded)).cpu()
    #recon = 1.0 * (th.sigmoid(decoded) > 0.5).cpu()
    #recon = th.sigmoid(decoded.exp()).cpu()
    recon = th.cat([lcd[:,None].cpu(), recon], 0)
    writer.add_image('reconstruction', utils.combine_imgs(recon, 2, 8)[None], epoch)
    #samples = self.sample(25)
    #writer.add_image('samples', utils.combine_imgs(samples, 5, 5)[None], epoch)

  def sample(self, n):
    import ipdb; ipdb.set_trace()
    prior_idxs = self.transformerCNN.sample(n)[0]
    prior_enc = self.vq.idx_to_encoding(prior_idxs)
    prior_enc = prior_enc.reshape([n, 7, 7, -1]).permute(0, 3, 1, 2)
    decoded = self.decoder(prior_enc)
    return 1.0*(decoded.exp() > 0.5).cpu()

class Encoder(nn.Module):
  def __init__(self, C):
    super().__init__()
    H = C.nfilter
    self.net = nn.Sequential(
        nn.Conv2d(1, H, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 2, padding=1),
        nn.ReLU(),
        nn.GroupNorm(32, H),
        nn.Conv2d(H, H, 3, 1, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, C.vqD, 1, 1),
    )
  def forward(self, x):
    return self.net(x[:,None])

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
  def __init__(self, C):
    super().__init__()
    H = C.nfilter
    self.net = nn.Sequential(
      nn.ConvTranspose2d(C.vqD, H, 1, 1),
      nn.ReLU(),
      nn.GroupNorm(32, H),
      nn.ConvTranspose2d(H, H, 4, 2, padding=2),
      #nn.ConvTranspose2d(C.vqD, H, 4, 2, padding=2),
      nn.ReLU(),
      nn.ConvTranspose2d(H, H, 3, 1),
      nn.ReLU(),
      nn.GroupNorm(32, H),
      nn.ConvTranspose2d(H, H, 3, 1, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(H, 1, 4, 2, padding=1),
    )
    
  def forward(self, x):
    return self.net(x)
