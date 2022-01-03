import copy
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
from research.nets.autoencoders.state_graph_ae import State_Graph_AE
from ._base import VideoModel

class State_Graph_FRNLD(VideoModel):
  def __init__(self, env, G):
    super().__init__(env, G)
    # <LOAD ronald>
    sd = th.load(G.weightdir / 'State_Graph_AE.pt', map_location=th.device(G.device))
    ronaldG = sd.pop('G')
    ronaldG.device = G.device
    self.ronald = State_Graph_AE(env, ronaldG)
    self.ronald.load(G.weightdir)
    for p in self.ronald.parameters():
      p.requires_grad = False
    self.ronald.eval()
    # </LOAD ronald>

    self.z_size = self.ronald.z_size
    self.n_objects = len(env.world_def.objects)

    self.block_size = self.G.window
    tfG = copy.deepcopy(G)
    tfG.n_embed = self.z_size
    self.graph_mixer = TransformerBlock(self.n_objects, tfG, causal=False)
    # GPT STUFF
    self.pos_emb = nn.Parameter(th.zeros(1, self.block_size, G.n_embed))
    self.cond_in = nn.Linear(self.act_n, G.n_embed // 2, bias=False)
    # input embedding stem
    self.embed = nn.Linear(self.z_size, G.n_embed // 2, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[TransformerBlock(self.block_size, G) for _ in range(G.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(G.n_embed)
    self.out_net = nn.Linear(G.n_embed, self.z_size)
    self._init()

  def forward(self, z, action):
    x = self.embed(z)
    # TODO: was figuring out how to reshape things to make this work
    # forward the GPT model
    BS, T, N, E = x.shape
    x = self.graph_mixer(x.flatten(0,1)).unflatten(0, [BS, T])
    x = x.swapaxes(1, 2).reshape([BS * N, T, E])
    # SHIFT RIGHT (add a padding on the left)
    x = th.cat([th.zeros(BS*N, 1, E).to(self.G.device), x[:, :-1]], dim=1)
    action = th.cat([th.zeros(BS, 1, action.shape[-1]).to(self.G.device), action[:, :-1]], dim=1)
    cin = self.cond_in(action)
    cin = cin.repeat_interleave(N, dim=0)
    if action.ndim == 2:
      x = th.cat([x, cin[:, None].repeat_interleave(self.block_size, 1)], -1)
    else:
      x = th.cat([x, cin], -1)
    x += self.pos_emb  # each position maps to a (learnable) vector
    x = self.blocks(x)
    logits = self.ln_f(x)
    x = self.out_net(logits)
    return x.unflatten(0, [BS, N]).swapaxes(1, 2)

  def loss(self, batch):
    z = self.ronald.encode(batch, noise=False).detach()
    logits = self.forward(z, batch['action'])
    loss = ((th.tanh(logits) - z)**2).mean()
    return loss, {'loss/total': loss}

  def onestep(self, batch, i, temp=1.0):
    z = self.ronald.encode(batch, noise=False)
    logits = self.forward(z, batch['action'])
    z_sample = self.ronald.vq(logits, noise=True)[0][:, i:i+1].reshape([-1, self.ronald.G.vqD])
    dist = self.ronald.decoder(z_sample)
    batch['full_state'][:, i] = dist['full_state'].mean
    return batch

  def latent_onestep(self, z, a, i, temp=1.0):
    logits = self.forward(z, a)
    z[:,i] = self.ronald.vq(logits, noise=True)[0][:,i]
    return z

  def latent_sample(self, z, a, start, temp):
    for i in range(start, self.block_size):
      z = self.latent_onestep(z, a, i, temp=temp)
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
      batch['full_state'] = th.zeros(n, self.block_size, self.full_state_n).to(self.G.device)
      start = 0
      if prompts is not None:
        batch['full_state'][:, :prompt_n] = prompts['full_state'][:, :prompt_n]
        start = prompt_n
      z = self.ronald.encode(batch, noise=False)
      z_sample = th.zeros(n, self.block_size, self.n_objects, self.ronald.G.vqD).to(self.G.device)
      z_sample[:, :prompt_n] = z[:, :prompt_n]
      # SAMPLE FORWARD IN LATENT SPACE, ACTION CONDITIONED
      z_sample = self.latent_sample(z_sample, action, start, temp)
      z_sample = z_sample.reshape([n * self.block_size, self.n_objects, self.ronald.G.vqD])

      # DECODE
      dist = self.ronald.decoder(z_sample)
      batch['full_state'] = (dist['full_state'].mean).reshape([n, self.block_size, -1])
    return batch
