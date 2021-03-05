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

class State_VQVAE(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.n_embed
    C.vqD = 128
    C.vqK = 256
    C.beta = 0.25
    # encoder -> VQ -> decoder
    state_n = env.observation_space['state'].shape[0]
    self.encoder = Encoder(state_n, C)
    self.vq = VectorQuantizer(C.vqK, C.vqD, C.beta, C)
    self.decoder = Decoder(state_n, C)
     ## prior. this is usually learned after the other stuff has been trained, but we do it all in one swoop.
    #self.transformerCNN = TransformerCNN(in_size=C.vqK, block_size=4*4, head='cat', C=C)
    #self.prior_optimizer = Adam(self.transformerCNN.parameters(), lr=C.prior_lr, betas=(0.5, 0.999))
    self.C = C

  def loss(self, x, eval=False):
    x = x['state']
    embed_loss, decoded, perplexity, idxs = self.forward(x)
    recon_loss = -tdib.Bernoulli(logits=decoded).log_prob(x).mean()
    loss = recon_loss + embed_loss
    prior_loss = th.zeros(1)
    # PRIOR
    #self.zero_grad()
    #code_idxs = F.one_hot(idxs.detach(), self.C.vqK).float().flatten(1,2)
    #dist = self.transformerCNN.forward(code_idxs)
    #prior_loss = -dist.log_prob(code_idxs).mean()
    #prior_loss.backward()
    #self.prior_optimizer.step()
    metrics = {'vq_vae_loss': loss, 'recon_loss': recon_loss, 'embed_loss': embed_loss, 'perplexity': perplexity, 'prior_loss': prior_loss}
    if eval:
      metrics['decoded'] = decoded
    return loss, metrics

  def forward(self, x):
    z_e = self.encoder(x)
    embed_loss, z_q, perplexity, idxs = self.vq(z_e)
    z_q = z_q.reshape(-1, self.C.vqD*8)
    decoded = self.decoder(z_q)
    return embed_loss, decoded, perplexity, idxs

  def sample(self, n):
    import ipdb; ipdb.set_trace()
    prior_idxs = self.transformerCNN.sample(n)[0]
    prior_enc = self.vq.idx_to_encoding(prior_idxs)
    prior_enc = prior_enc.reshape([n, 7, 7, -1]).permute(0, 3, 1, 2)
    decoded = self.decoder(prior_enc)
    return 1.0*(decoded.exp() > 0.5).cpu()

class VectorQuantizer(nn.Module):
  """from: https://github.com/MishaLaskin/vqvae"""
  def __init__(self, K, D, beta, C):
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