from re import I
import yaml
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from research.nets.common import TransformerBlock, BinaryHead
from research import utils
from research.nets.autoencoders.rnlda import RNLDA
from ._base import VideoModel

class FlatRonald(VideoModel):
  def __init__(self, env, G):
    super().__init__(env, G)
    # <LOAD ronald>
    sd = th.load(G.weightdir / 'RNLDA.pt')
    ronaldG = sd.pop('G')
    self.ronald = RNLDA(env, ronaldG)
    self.ronald.load(G.weightdir)
    for p in self.ronald.parameters():
      p.requires_grad = False
    self.ronald.eval()
    # </LOAD ronald>

    self.zW = int(self.ronald.G.wh_ratio*4)
    self.size = self.ronald.G.vqD * 4 * self.zW
    self.block_size = self.G.window
    # GPT STUFF
    self.pos_emb = nn.Parameter(th.zeros(1, self.block_size, G.n_embed))
    self.cond_in = nn.Linear(self.act_n, G.n_embed // 2, bias=False)
    # input embedding stem
    self.embed = nn.Linear(self.size, G.n_embed // 2, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[TransformerBlock(self.block_size, G) for _ in range(G.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(G.n_embed)
    self.out_net = nn.Linear(G.n_embed, self.size)
    self._init()

  def forward(self, z, action):
    x = self.embed(z)
    # forward the GPT model
    BS, T, E = x.shape
    # SHIFT RIGHT (add a padding on the left)
    x = th.cat([th.zeros(BS, 1, E).to(self.G.device), x[:, :-1]], dim=1)
    action = th.cat([th.zeros(BS, 1, action.shape[-1]).to(self.G.device), action[:, :-1]], dim=1)
    cin = self.cond_in(action)
    if action.ndim == 2:
      x = th.cat([x, cin[:, None].repeat_interleave(self.block_size, 1)], -1)
    else:
      x = th.cat([x, cin], -1)
    x += self.pos_emb  # each position maps to a (learnable) vector
    x = self.blocks(x)
    logits = self.ln_f(x)
    return self.out_net(logits)

  def loss(self, batch):
    z = self.ronald.encode(batch, noise=False).detach()
    logits = self.forward(z, batch['action'])
    loss = ((th.tanh(logits) - z)**2).mean()
    return loss, {'loss/total': loss}

  def onestep(self, batch, i, temp=1.0):
    logits = self.forward(batch)
    import ipdb; ipdb.set_trace()
    sample = dist.sample()
    batch['lcd'][:, i] = sample[:, i, :self.G.imsize].reshape(batch['lcd'][:, i].shape)
    proprio_code = sample[:, i, self.G.imsize:]
    proprio = self.ronald.decoder(proprio_code).mean
    batch['proprio'][:, i] = proprio
    return batch

  def latent_onestep(self, z, a, i, temp=1.0):
    logits = self.forward(z, a)
    z[:,i] = self.ronald.vq(logits, noise=True)[0][:,i]
    return z

  def latent_sample(self, z, a, start):
    for i in range(start, self.block_size):
      z = self.latent_onestep(z, a, i, temp=1.0)
    return z

  def sample(self, n, action=None, prompts=None, prompt_n=10, temp=1.0):
    # TODO: feed act_n
    with th.no_grad():
      # CREATE PROMPT
      if action is None:
        action = (th.rand(n, self.block_size, self.act_n) * 2 - 1).to(self.G.device)
      else:
        n = action.shape[0]
      batch = {}
      batch['lcd'] = th.zeros(n, self.block_size, self.G.lcd_h, self.G.lcd_w).to(self.G.device)
      batch['proprio'] = th.zeros(n, self.block_size, self.proprio_n).to(self.G.device)
      start = 0
      if prompts is not None:
        batch['lcd'][:, :prompt_n] = prompts['lcd'][:, :prompt_n]
        batch['proprio'][:, :prompt_n] = prompts['proprio'][:, :prompt_n]
        start = prompt_n
      z = self.ronald.encode(batch, noise=False)
      z_sample = th.zeros(n, self.block_size, self.ronald.G.vqD * 4 * self.zW).to(self.G.device)
      z_sample[:, :prompt_n] = z[:, :prompt_n]
      # SAMPLE FORWARD IN LATENT SPACE, ACTION CONDITIONED
      z_sample = self.latent_sample(z_sample, action, start)
      z_sample = z_sample.reshape([n * self.block_size, self.ronald.G.vqD, 4, self.zW])

      # DECODE
      dist = self.ronald.decoder(z_sample)
      batch['lcd'] = (1.0 * (dist['lcd'].probs > 0.5)).reshape([n, self.block_size, 1, self.G.lcd_h, self.G.lcd_w])
      batch['proprio'] = (dist['proprio'].mean).reshape([n, self.block_size, -1])
    return batch
