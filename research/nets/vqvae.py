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
from gms import utils
from gms.autoregs.transformer import TransformerCNN
from .vq import VectorQuantizer

class VQVAE(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    # encoder -> VQ -> decoder
    self.encoder = Encoder(C)
    self.vq = VectorQuantizer(C.vqK, C.vqD, C.beta, C)
    self.decoder = Decoder(C)

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
    H = C.hidden_size
    self.net = nn.Sequential(
        nn.Conv2d(1, H, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, 1, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, C.vqD, 3, 1, padding=1),
        nn.ReLU(),
    )
  def forward(self, x):
    return self.net(x)

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
    H = C.hidden_size

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
    return self.net(x)
