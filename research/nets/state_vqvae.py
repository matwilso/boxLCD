import sys
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead, aggregate, MultiHead, ConvEmbed
import torch as th
from torch import distributions as tdib
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gms import utils
from gms.autoregs.transformer import TransformerCNN
from .vqvae import VectorQuantizer

class State_VQVAE(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.n_embed
    C.vqD = 128
    C.vqK = 256
    C.beta = 0.25
    # encoder -> VQ -> decoder
    self.state_n = env.observation_space['state'].shape[0]
    self.encoder = Encoder(self.state_n, C)
    self.vq = VectorQuantizer(C.vqK, C.vqD, C.beta, C)
    self.decoder = Decoder(self.state_n, C)
    self.C = C

  def loss(self, x, eval=False):
    x = x['state']
    embed_loss, dec_dist, perplexity, idxs = self.forward(x)
    recon_loss = -dec_dist.log_prob(x).mean()
    loss = recon_loss + embed_loss
    metrics = {'vq_vae_loss': loss, 'recon_loss': recon_loss, 'embed_loss': embed_loss, 'perplexity': perplexity}
    if eval:
      metrics['decoded'] = dec_dist
    return loss, metrics

  def forward(self, x):
    z_e = self.encoder(x)
    embed_loss, z_q, perplexity, idxs = self.vq(z_e)
    z_q = z_q.reshape(-1, self.C.vqD*8)
    decoded = self.decoder(z_q)
    dec_dist = tdib.Normal(decoded, 1)
    return embed_loss, dec_dist, perplexity, idxs

class Encoder(nn.Module):
  def __init__(self, in_size, C):
    super().__init__()
    H = C.n_embed
    self.net = nn.Sequential(
      nn.Linear(in_size, H),
      nn.ReLU(),
      nn.Linear(H, H),
      nn.LayerNorm(H),
      nn.ReLU(),
      nn.Linear(H, 8*C.vqD),
      nn.ReLU(),
      nn.Unflatten(-1, (8,C.vqD)),
      nn.Linear(C.vqD, C.vqD),
    )
  def forward(self, x):
    return self.net(x)

class Decoder(nn.Module):
  def __init__(self, out_size, C):
    super().__init__()
    H = C.n_embed
    self.net = nn.Sequential(
       nn.Linear(8*C.vqD, H),
       nn.ReLU(),
       nn.Linear(H, H),
       nn.ReLU(),
       nn.Linear(H, out_size),
    )
  def forward(self, x):
    return self.net(x)