import matplotlib.pyplot as plt
import torch as torchvision
from torch.optim import Adam
from itertools import chain, count
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
#from nets.common import GaussHead, MDNHead, CausalSelfAttention, TransformerBlock, BinaryHead, aggregate, MultiHead, ConvEmbed
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils

# TODO: something like binaryquantize, but having 3-4 values. so somehow you need to split the real number line up
# basically so that you can do multi-modal stuff in a single vector

class BinaryQuantize(nn.Module):
  def __init__(self, num_hiddens, n_embed):
    super().__init__()
    self.n_embed = n_embed
    self.proj = nn.Linear(num_hiddens, n_embed)

  def forward(self, z):
    #logits = self.proj(z)
    dist = thd.Bernoulli(logits=z)
    z_q = dist.sample()
    z_q += dist.probs - dist.probs.detach()
    entropy = dist.entropy().mean()
    return z_q, entropy, dist.probs

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