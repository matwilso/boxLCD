from re import I
import numpy as np
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from research import utils
from research.nets.common import ResBlock
from torch.optim import Adam
from ._base import Autoencoder, SingleStepAE, MultiStepAE
from jax.tree_util import tree_map

class MultiStepArbiter(MultiStepAE):
  def __init__(self, env, G):
    super().__init__(env, G)
    self.z_size = 256
    state_n = env.observation_space.spaces['proprio'].shape[0]
    act_n = env.action_space.shape[0]
    self.encoder = Encoder(state_n, self.z_size, G)
    self.decoder = Decoder(act_n, state_n, self.z_size, G)
    self._init()

  def _unprompted_eval(self, epoch=None, writer=None, metrics=None, batch=None, arbiter=None):
    return {}

  def save(self, dir, batch):
    print("SAVED MODEL", dir)
    path = dir / f'{self.name}.pt'
    self.eval()
    self.encoder.eval()

    class TracedArbiter(nn.Module):
      def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
      def forward(self, batch):
        z = self.encoder(batch)
        dec = self.decoder(z)
        return z, dec[2]
    d = TracedArbiter(self.encoder, self.decoder)
    jit_enc = th.jit.trace(d, self.batch_proc(batch))
    th.jit.save(jit_enc, str(path))
    print(path)

  def loss(self, batch):
    z = self.encoder(batch)
    decoded = self.decoder.forward_dist(z)
    recon_losses = {}
    recon_losses['loss/recon_proprio'] = -decoded['proprio'].log_prob(batch['proprio']).mean()
    recon_losses['loss/recon_lcd'] = -decoded['lcd'].log_prob(batch['lcd']).mean()
    recon_losses['loss/recon_action'] = -decoded['action'].log_prob(batch['action'][:, :-1]).mean()
    recon_loss = sum(recon_losses.values())
    metrics = {'loss/recon_total': recon_loss, **recon_losses}
    return recon_loss, metrics

  def encode(self, batch, flatten=None):
    return self.encoder(batch)

  def _decode(self, z):
    return self.decoder.forward_dist(z)

class Encoder(nn.Module):
  def __init__(self, state_n, out_size, G):
    super().__init__()
    n = G.hidden_size
    self.state_embed = nn.Sequential(
        nn.Linear(state_n, n),
        nn.ReLU(),
        nn.Linear(n, n),
        nn.ReLU(),
        nn.Flatten(-2),
        nn.Linear(G.window * n, n),
    )
    nf = G.nfilter
    size = (G.lcd_h * G.lcd_w) // 64
    self.seq = nn.ModuleList([
        nn.Conv2d(G.window, nf, 3, 2, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Conv2d(nf, nf, 3, 2, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Conv2d(nf, nf, 3, 2, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Flatten(-3),
        nn.Linear(size * nf, out_size),

    ])

  def forward(self, batch):
    state = batch['proprio']
    lcd = batch['lcd']
    emb = self.state_embed(state)
    x = lcd
    for layer in self.seq:
      if isinstance(layer, ResBlock):
        x = layer(x, emb)
      else:
        x = layer(x)
    return x

class Decoder(nn.Module):
  def __init__(self, act_n, state_n, in_size, G):
    super().__init__()
    #assert G.lcd_h == 16 and G.lcd_w == 32
    if G.lcd_w == 32:
      W = 4
    elif G.lcd_w == 24:
      W = 3
    elif G.lcd_w == 16:
      W = 2
    nf = G.nfilter
    self.net = nn.Sequential(
        nn.ConvTranspose2d(in_size, nf, (2, W), 2),
        nn.ReLU(),
        nn.ConvTranspose2d(nf, nf, 4, 4, padding=0),
        nn.ReLU(),
        nn.Conv2d(nf, nf, 3, 1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(nf, G.window, 4, 2, padding=1),
    )
    n = G.hidden_size
    self.state_net = nn.Sequential(
        nn.Linear(in_size, n),
        nn.ReLU(),
        nn.Linear(n, G.window * n),
        nn.ReLU(),
        utils.Reshape(-1, G.window, n),
        nn.Linear(n, state_n),
    )
    self.act_net = nn.Sequential(
        nn.Linear(in_size, n),
        nn.ReLU(),
        nn.Linear(n, (G.window - 1) * n),
        nn.ReLU(),
        utils.Reshape(-1, G.window-1, n),
        nn.Linear(n, act_n),
    )

  def forward_dist(self, x):
    lcd, proprio, act = self.forward(x)
    lcd_dist = thd.Bernoulli(logits=lcd)
    state_dist = thd.Normal(proprio, 1)
    act_dist = thd.Normal(act, 1)
    return {'lcd': lcd_dist, 'proprio': state_dist, 'action': act_dist}

  def forward(self, x):
    lcd = self.net(x[..., None, None])
    proprio = self.state_net(x)
    act = self.act_net(x)
    return lcd, proprio, act