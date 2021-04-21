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
from research.nets.common import GaussHead, MDNHead, CausalSelfAttention, TransformerBlock, BinaryHead, aggregate, MultiHead, ConvEmbed, ConvBinHead, ResBlock
from research import utils
import ignite
from ._base import VideoModel
from jax.tree_util import tree_map, tree_multimap

class BasicFF(VideoModel):
  def __init__(self, env, G):
    super().__init__(env, G)
    self.encoder = Encoder(env, G)
    self.decoder = Decoder(env, G)
    self.zH = 4
    self.zW = int(G.wh_ratio * self.zH)
    self.z_size = self.zH * self.zW * G.vqD
    self._init()

  def loss(self, batch):
    # autoencode
    z = self.encoder(batch)
    decoded = self.decoder(z)
    import ipdb; ipdb.set_trace()
    # compute losses
    recon_losses = {}
    recon_losses['loss/recon_proprio'] = -decoded['proprio'].log_prob(batch['proprio']).mean()
    recon_losses['loss/recon_lcd'] = -decoded['lcd'].log_prob(batch['lcd'][:, None]).mean()
    loss = recon_loss = sum(recon_losses.values())
    metrics = {'loss/total': loss, **recon_losses, 'loss/recon_total': recon_loss}
    return loss, metrics

  def sample(self, n, action=None, prompts=None, prompt_n=10):
    with th.no_grad():
      if action is not None:
        n = action.shape[0]
      else:
        action = (th.rand(n, self.G.window, self.act_n) * 2 - 1).to(self.G.device)
      if prompts is None:
        import ipdb; ipdb.set_trace()
        prior = self.imagine(action)
        feat = self.get_feat(prior)
        decoded = self.decoder(feat.flatten(0, 1))
        batch = {'lcd': (1.0 * (decoded['lcd'].probs > 0.5)), 'proprio': decoded['proprio'].mean}
        batch['lcd'] = batch['lcd'].reshape(n, -1, 1, self.G.lcd_h, self.G.lcd_w)
        batch['proprio'] = th.zeros([*batch['lcd'].shape[:2], self.env.observation_space['proprio'].shape[0]]).to(batch['lcd'].device)
      else:
        import ipdb; ipdb.set_trace()
        batch = tree_map(lambda x: x[:, :prompt_n], prompts)
        flat_batch = tree_map(lambda x: x.flatten(0, 1), batch)
        embed = self.encoder(flat_batch).unflatten(0, (*batch['lcd'].shape[:2],))
        post, prior = self.observe(embed, action[:, :prompt_n])
        prior = self.imagine(action[:, prompt_n:], state=tree_map(lambda x: x[:, -1], post))
        feat = self.get_feat(prior)
        decoded = self.decoder(feat.flatten(0, 1))
        batch = {'lcd': (1.0 * (decoded['lcd'].probs > 0.5)), 'proprio': decoded['proprio'].mean}
        batch['lcd'] = batch['lcd'].reshape(n, -1, 1, self.G.lcd_h, self.G.lcd_w)
        batch['proprio'] = th.zeros([*batch['lcd'].shape[:2], self.env.observation_space['proprio'].shape[0]]).to(batch['lcd'].device)
        prompts['lcd'] = prompts['lcd'][:, :, None]
        batch = tree_multimap(lambda x, y: th.cat([x, y[:, :prompt_n]], 1), batch, utils.subdict(prompts, ['lcd', 'proprio']))
        prompts['lcd'] = prompts['lcd'][:, :, 0]
    return batch

  def get_feat(self, state):
    return th.cat([state['stoch'], state['deter']], -1)

  def get_dist(self, state):
    return thd.Normal(state['mean'], state['std'])
    # return thd.MultivariateNormal(state['mean'], scale_tril=th.diag_embed(state['std']))

class Encoder(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    state_n = env.observation_space.spaces['proprio'].shape[0]
    act_n = env.action_space.shape[0]
    n = G.hidden_size
    self.sa_embed = nn.Sequential(
        nn.Linear(act_n + state_n, n),
        nn.ReLU(),
        nn.Linear(n, n),
        nn.ReLU(),
        nn.Linear(n, n),
    )
    nf = G.nfilter
    self.seq = nn.ModuleList([
        nn.Conv2d(1, nf, 3, 1, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Conv2d(nf, nf, 3, 2, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Conv2d(nf, nf, 3, 2, padding=1),
        ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
        nn.Conv2d(nf, G.vqD, 1, 1),
    ])

  def forward(self, batch):
    state = batch['proprio']
    lcd = batch['lcd']
    act = batch['action']
    emb = self.sa_embed(th.cat([state, act], -1))
    x = lcd[:, None]
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
  def forward(self, x):
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = self.conv(x)
    return x

class Decoder(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    state_n = env.observation_space.spaces['proprio'].shape[0]
    n = G.hidden_size
    H = 4
    W = int(G.wh_ratio * H)
    self.state_net = nn.Sequential(
        nn.Flatten(-3),
        nn.Linear(G.vqD * H * W, n),
        nn.ReLU(),
        nn.Linear(n, n),
        nn.ReLU(),
        nn.Linear(n, state_n),
    )
    nf = G.nfilter
    self.net = nn.Sequential(
        Upsample(G.vqD, nf),
        nn.ReLU(),
        Upsample(nf, nf),
        nn.ReLU(),
        nn.Conv2d(nf, nf, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(nf, 1, 3, padding=1),
    )

  def forward(self, x):
    lcd_dist = thd.Bernoulli(logits=self.net(x))
    state_dist = thd.Normal(self.state_net(x), 1)
    return {'lcd': lcd_dist, 'proprio': state_dist}
