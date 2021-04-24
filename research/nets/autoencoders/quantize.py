import matplotlib.pyplot as plt
import torch as torchvision
from torch.optim import Adam
from itertools import chain, count
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
#from research.nets.common import GaussHead, MDNHead, CausalSelfAttention, TransformerBlock, BinaryHead, aggregate, MultiHead, ConvEmbed
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TanhD(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, z, noise=True):
    z = th.tanh(z)
    return z

class RNLD(nn.Module):
  """
  Real Number Line Discretization (RNLD)
  you can call it ronald

  Chunk up the number line into 4 bins.
  Assign all values in each bin to the mean of that bin.

  -1          -0.5            0             0.5         1.0
  | -----a------|------b------|------c-------|------d-----|
  """
  def __init__(self, num_cat, noise_level=0.25):
    super().__init__()
    self.num_cat = num_cat
    self.noise_level = noise_level

  def forward(self, z, noise):
    z = th.tanh(z)
    #zn = z + noise * (2 * th.rand(z.shape).to(z.device) - 1)
    if noise:
      zn = z + self.noise_level * (2 * th.rand(z.shape).to(z.device) - 1)
    else:
      zn = z

    z_q = -0.75 * (zn < -0.5) + \
        -0.25 * (th.logical_and(zn >= -0.5, zn < 0.0)) + \
        0.25 * (th.logical_and(zn >= 0.0, zn < 0.5)) + \
        0.75 * (zn >= 0.5)
    #z_q += zn - zn.detach()
    z_q += z - z.detach()

    idxs = 0 * (zn < -0.5) + \
        1 * (th.logical_and(zn >= -0.5, zn < 0.0)) + \
        2 * (th.logical_and(zn >= 0.0, zn < 0.5)) + \
        3 * (zn >= 0.5)
    return z_q, idxs

class BinaryQuantize(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, z, noise=True):
    #logits = self.proj(z)
    dist = thd.Bernoulli(logits=z)
    z_q = dist.sample()
    z_q += dist.probs - dist.probs.detach()
    entropy = dist.entropy().mean()
    if noise:
      return z_q, entropy, dist.probs
    else:
      return 1.0 * (dist.probs > 0.5), entropy, dist.probs  # deterministic mode

class VectorQuantizer(nn.Module):
  def __init__(self, K, D, beta, G):
    super().__init__()
    self.K = K
    self.D = D
    self.beta = beta
    self.embedding = nn.Embedding(self.K, self.D)
    self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

  def idx_to_encoding(self, one_hots):
    z_q = th.matmul(one_hots, self.embedding.weight)
    return z_q

  def forward(self, z):
    if z.ndim == 4:
      # reshape z -> (batch, height, width, channel) and flatten
      z = z.permute(0, 2, 3, 1).contiguous()
    z_flattened = z.view(-1, self.D)
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = th.sum(z_flattened ** 2, dim=1, keepdim=True) + th.sum(self.embedding.weight**2, dim=1) - 2 * th.matmul(z_flattened, self.embedding.weight.t())
    # find closest encodings
    min_encoding_indices = th.argmin(d, dim=1).unsqueeze(1)
    min_encodings = th.zeros(min_encoding_indices.shape[0], self.K).to(z.device)
    min_encodings.scatter_(1, min_encoding_indices, 1)
    # get quantized latent vectors
    z_q = th.matmul(min_encodings, self.embedding.weight).view(z.shape)
    # compute loss for embedding
    loss = th.mean((z_q.detach() - z)**2) + self.beta * th.mean((z_q - z.detach()) ** 2)
    # preserve gradients
    z_q = z + (z_q - z).detach()
    # perplexity
    e_mean = th.mean(min_encodings, dim=0)
    perplexity = th.exp(-th.sum(e_mean * th.log(e_mean + 1e-10)))
    # reshape back to match original input shape
    if z.ndim == 4:
      z_q = z_q.permute(0, 3, 1, 2).contiguous()
    return loss, z_q, perplexity, min_encoding_indices.view(z.shape[:-1])
