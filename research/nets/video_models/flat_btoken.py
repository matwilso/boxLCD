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
from nets.common import TransformerBlock, BinaryHead
import utils
from research.nets.autoencoders.bvae import BVAE
from ._base import VideoModel

class FlatBToken(VideoModel):
  def __init__(self, env, G):
    super().__init__(env, G)
    # <LOAD BVAE>
    sd = th.load(G.weightdir / 'BVAE.pt')
    bvaeC = sd.pop('G')
    self.bvae = BVAE(env, bvaeC)
    self.bvae.load(G.weightdir)
    for p in self.bvae.parameters():
      p.requires_grad = False
    self.bvae.eval()
    # </LOAD BVAE>

    self.size = self.bvae.G.vqD * 4 * 8
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
    self.dist_head = BinaryHead(G.n_embed, self.size, G)
    self._init()

  def forward(self, z, acts):
    x = self.embed(z)
    # forward the GPT model
    BS, T, E = x.shape
    # SHIFT RIGHT (add a padding on the left)
    x = th.cat([th.zeros(BS, 1, E).to(self.G.device), x[:, :-1]], dim=1)
    acts = th.cat([th.zeros(BS, 1, acts.shape[-1]).to(self.G.device), acts[:, :-1]], dim=1)
    cin = self.cond_in(acts)
    if acts.ndim == 2:
      x = th.cat([x, cin[:, None].repeat_interleave(self.block_size, 1)], -1)
    else:
      x = th.cat([x, cin], -1)
    x += self.pos_emb  # each position maps to a (learnable) vector
    x = self.blocks(x)
    logits = self.ln_f(x)
    return logits

  def loss(self, batch):
    z = self.bvae.encode(batch).detach()
    logits = self.forward(z, batch['acts'])
    dist = self.dist_head(logits)
    loss = -dist.log_prob(z).mean()
    return loss, {'loss/total': loss}

  def onestep(self, batch, i, temp=1.0):
    import ipdb; ipdb.set_trace()
    logits = self.forward(batch)
    dist = self.dist_head(logits / temp)
    sample = dist.sample()
    batch['lcd'][:, i] = sample[:, i, :self.G.imsize].reshape(batch['lcd'][:, i].shape)
    pstate_code = sample[:, i, self.G.imsize:]
    pstate = self.bvae.decoder(pstate_code).mean
    batch['pstate'][:, i] = pstate
    return batch

  def latent_onestep(self, z, a, i, temp=1.0):
    logits = self.forward(z, a)
    dist = self.dist_head(logits / temp)
    z[:, i] = dist.sample()[:, i]
    return z

  def latent_sample(self, z, a, start):
    for i in range(start, self.block_size):
      z = self.latent_onestep(z, a, i, temp=1.0)
    return z

  def sample(self, n, acts=None, prompts=None, prompt_n=10, temp=1.0):
    # TODO: feed act_n
    with th.no_grad():
      # CREATE PROMPT
      if acts is None:
        acts = (th.rand(n, self.block_size, self.act_n) * 2 - 1).to(self.G.device)
      else:
        n = acts.shape[0]
      batch = {}
      batch['lcd'] = th.zeros(n, self.block_size, self.G.lcd_h, self.G.lcd_w).to(self.G.device)
      batch['pstate'] = th.zeros(n, self.block_size, self.pstate_n).to(self.G.device)
      start = 0
      if prompts is not None:
        batch['lcd'][:, :prompt_n] = prompts['lcd'][:, :prompt_n]
        batch['pstate'][:, :prompt_n] = prompts['pstate'][:, :prompt_n]
        start = prompt_n
      z = self.encode(batch)
      z_sample = th.zeros(n, self.block_size, self.bvae.G.vqD * 4 * 8).to(self.G.device)
      z_sample[:, :prompt_n] = z[:, :prompt_n]
      z_sample = z_sample.reshape([n * self.block_size, self.bvae.G.vqD, 4, 8])

      # SAMPLE FORWARD IN LATENT SPACE, ACTION CONDITIONED
      z_sample = self.latent_sample(z_sample, acts, start)

      # DECODE
      dist = self.bvae.decoder(z_sample)
      batch['lcd'] = (1.0 * (dist['lcd'].probs > 0.5)).reshape([n, self.block_size, 1, self.G.lcd_h, self.G.lcd_w])
      batch['pstate'] = (dist['pstate'].mean).reshape([n, self.block_size, -1])
    return batch