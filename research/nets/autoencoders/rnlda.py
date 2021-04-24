import numpy as np
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from research import utils
from research.nets.common import ResBlock
from .quantize import BinaryQuantize, RNLD
from ._base import Autoencoder, SingleStepAE
from .bvae import Encoder, Decoder

class RNLDA(SingleStepAE):
  """Real Number Line Discrete Autoencoder. ronalda"""
  def __init__(self, env, G):
    super().__init__(env, G)
    # encoder -> binary -> decoder
    self.encoder = Encoder(env, G)
    self.vq = RNLD(4)
    self.decoder = Decoder(env, G)
    self.zH = 4
    self.zW = int(G.wh_ratio * self.zH)
    self.z_size = self.zH * self.zW * G.vqD
    self._init()

  def sample_z(self, n):
    z = th.bernoulli(0.5 * th.ones(n, self.z_size)).to(self.G.device).reshape([n, -1, self.zH, self.zW])
    return z

  def loss(self, batch):
    # autoencode
    z_e = self.encoder(batch)
    z_q, idxs = self.vq(z_e)
    decoded = self.decoder(z_q)
    # compute losses
    recon_losses = {}
    recon_losses['loss/recon_proprio'] = -decoded['proprio'].log_prob(batch['proprio']).mean()
    recon_losses['loss/recon_lcd'] = -decoded['lcd'].log_prob(batch['lcd'][:, None]).mean()
    recon_loss = sum(recon_losses.values())
    loss = recon_loss
    metrics = {'loss/total': loss, **recon_losses, 'loss/recon_total': recon_loss,
               'idx0_frac': th.mean(1.0 * (idxs == 0)),
               'idx1_frac': th.mean(1.0 * (idxs == 1)),
               'idx2_frac': th.mean(1.0 * (idxs == 2)),
               'idx3_frac': th.mean(1.0 * (idxs == 3))
               }
    return loss, metrics

  def encode(self, batch, noise, flatten=True):
    shape = batch['lcd'].shape
    if len(shape) == 4:
      batch = {key: val.clone().flatten(0, 1) for key, val in batch.items()}
    batch['lcd'].reshape
    z_e = self.encoder(batch)
    # return z_e.flatten(-3)
    z_q, idxs = self.vq(z_e, noise=noise)
    if flatten:
      z_q = z_q.flatten(-3)
      assert z_q.shape[-1] == self.z_size, 'encode shape should equal the z_size. probably forgot to change one.'
    if len(shape) == 4:
      return z_q.reshape([*shape[:2], *z_q.shape[1:]])
    return z_q

  def _decode(self, z_q):
    decoded = self.decoder(z_q)
    return decoded
