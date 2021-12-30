import numpy as np
import torch as th
from torch import distributions as thd
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from research import utils
from research.nets.common import ResBlock
from research.nets.quantize import BinaryQuantize
from research.nets.quantize import RNLD
from ._base import Autoencoder, SingleStepAE

# TODO: save this and other models like the arbiter. where we don't have to worry about settings. we just load the compiled version

class State_AE(SingleStepAE):
  def __init__(self, env, G):
    super().__init__(env, G)
    # encoder -> binary -> decoder
    self.encoder = StateEncoder(env, G)
    #self.vq = BinaryQuantize()
    self.vq = RNLD(4)
    self.decoder = StateDecoder(env, G)
    self.optimizer = Adam(self.parameters(), lr=self.G.lr)
    self.z_size = self.G.vqD

  def sample_z(self, n):
    z = th.rand(n, self.z_size).to(self.G.device) * 2 - 1
    return z

  def loss(self, batch):
    # autoencode
    z_e = self.encoder(batch)
    z_q, idxs = self.vq(z_e, noise=True)
    decoded = self.decoder(z_q)
    # compute losses
    recon_losses = {}
    recon_losses['loss/recon_full_state'] = -decoded['full_state'].log_prob(batch['full_state']).mean()
    recon_loss = sum(recon_losses.values())
    loss = recon_loss
    metrics = {'loss/total': loss, **recon_losses, 'loss/recon_total': recon_loss,
               'idx0_frac': th.mean(1.0 * (idxs == 0)),
               'idx1_frac': th.mean(1.0 * (idxs == 1)),
               'idx2_frac': th.mean(1.0 * (idxs == 2)),
               'idx3_frac': th.mean(1.0 * (idxs == 3))
               }
    return loss, metrics

  def encode(self, batch, noise, quantize=True):
    shape = batch['full_state'].shape
    if len(shape) == 3:
      batch = {key: val.clone().flatten(0, 1) for key, val in batch.items()}
    z_e = self.encoder(batch)
    if quantize:
      z_q, idxs = self.vq(z_e, noise=noise)
    else:
      z_q = z_e
    if len(shape) == 3:
      return z_q.reshape([*shape[:2], *z_q.shape[1:]])
    return z_q

  def _decode(self, z_q):
    decoded = self.decoder(z_q)
    return decoded

class StateEncoder(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    H = G.hidden_size
    state_n = env.observation_space.spaces['full_state'].shape[0]
    self.state_embed = nn.Sequential(
        nn.Linear(state_n, H),
        nn.ReLU(),
        nn.Linear(H, H),
        nn.ReLU(),
        nn.Linear(H, G.vqD),
    )
  def forward(self, batch):
    state = batch['full_state']
    x = self.state_embed(state)
    return x

class StateDecoder(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    n = G.hidden_size
    state_n = env.observation_space.spaces['full_state'].shape[0]
    self.state_net = nn.Sequential(
        nn.Linear(G.vqD, n),
        nn.ReLU(),
        nn.Linear(n, n),
        nn.ReLU(),
        nn.Linear(n, state_n),
    )
  def forward(self, x):
    return {'full_state': thd.Normal(self.state_net(x), 1)}