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
from nets.common import TransformerBlock, BinaryHead
import utils
#from nets.statevq import SVAE

class FlatEverything(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    self.G = G
    self.env = env
    self.act_n = env.action_space.shape[0]
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    self.proprio_n = env.observation_space.spaces['proprio'].shape[0]
    self.block_size = self.G.window

    # LOAD SVAE
    sd = th.load(G.weightdir/'svae.pt')
    svaeC = sd.pop('G')
    self.svae = SVAE(env, svaeC)
    self.svae.load(G.weightdir)
    for p in self.svae.parameters():
      p.requires_grad = False
    self.svae.eval()
    # </LOAD SVAE>

    self.size = self.G.imsize + G.vqK
    self.gpt_size = self.G.imsize + G.vqK
    self.dist = self.G.decode
    self.block_size = self.G.window

    self.pos_emb = nn.Parameter(th.zeros(1, self.block_size, G.n_embed))
    self.cond_in = nn.Linear(self.act_n, G.n_embed // 2, bias=False)
    # input embedding stem
    self.embed = nn.Linear(self.gpt_size, G.n_embed // 2, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[TransformerBlock(self.block_size, G) for _ in range(G.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(G.n_embed)
    self.dist_head = BinaryHead(G.n_embed, self.gpt_size, G)
    self.optimizer = Adam(self.parameters(), lr=G.lr)
    self.to(G.device)

  def save(self, dir):
    print("SAVED MODEL", dir)
    path = dir / 'flatev.pt'
    sd = self.state_dict()
    sd['G'] = self.G
    th.save(sd, path)
    print(path)
    self.svae.save(dir)

  def load(self, dir):
    path = dir / 'flatev.pt'
    sd = th.load(path)
    G = sd.pop('G')
    self.load_state_dict(sd)
    self.svae.load(dir)
    print(f'LOADED {path}')

  def forward(self, batch):
    BS, EPL, *HW = batch['lcd'].shape
    lcd = batch['lcd'].reshape(BS, EPL, np.prod(HW))
    acts = batch['acts']
    z_q = self.svae(utils.filtdict(batch, 'proprio'))[0]
    x = th.cat([lcd, z_q], -1)
    # forward the GPT model
    x = self.embed(x)
    BS, T, E = x.shape
    # SHIFT RIGHT (add a padding on the left)
    x = th.cat([th.zeros(BS, 1, E).to(self.G.device), x[:, :-1]], dim=1)
    acts = th.cat([th.zeros(BS, 1, acts.shape[-1]).to(self.G.device), acts[:,:-1]], dim=1)
    cin = self.cond_in(acts)
    if acts.ndim == 2:
      x = th.cat([x, cin[:, None].repeat_interleave(self.block_size, 1)], -1)
    else:
      x = th.cat([x, cin], -1)
    x += self.pos_emb  # each position maps to a (learnable) vector
    x = self.blocks(x)
    logits = self.ln_f(x)
    return logits

  def train_step(self, batch, dry=False):
    self.optimizer.zero_grad()
    loss, metrics = self.loss(batch)
    if not dry:
      loss.backward()
      self.optimizer.step()
    return metrics

  def loss(self, batch):
    BS, EPL, *HW = batch['lcd'].shape
    metrics = {}
    logits = self.forward(batch)
    dist = self.dist_head(logits)

    target_lcd = batch['lcd'].reshape(BS, EPL, np.prod(HW))
    target_proprio = self.svae(utils.filtdict(batch, 'proprio'))[0].detach()
    target = th.cat([target_lcd, target_proprio], -1)
    loss = -dist.log_prob(target).mean([0, 1])
    metrics['loss/lcd'] = loss[:self.G.imsize].mean()
    metrics['loss/state'] = loss[self.G.imsize:].mean()
    metrics['loss/total'] = total_loss = loss.mean()
    return total_loss, metrics

  def onestep(self, batch, i, temp=1.0):
    logits = self.forward(batch)
    dist = self.dist_head(logits / temp)
    sample = dist.sample()
    batch['lcd'][:, i] = sample[:, i, :self.G.imsize].reshape(batch['lcd'][:,i].shape)
    proprio_code = sample[:, i, self.G.imsize:]
    proprio = self.svae.decoder(proprio_code).mean
    batch['proprio'][:, i] = proprio
    return batch

  def sample(self, n, acts=None, prompts=None):
    # TODO: feed act_n
    with th.no_grad():
      if acts is not None:
        n = acts.shape[0]
      batch = {}
      batch['lcd'] = th.zeros(n, self.block_size, self.G.imsize).to(self.G.device)
      batch['proprio'] = th.zeros(n, self.block_size, self.proprio_n).to(self.G.device)
      batch['acts'] = acts if acts is not None else (th.rand(n, self.block_size, self.act_n) * 2 - 1).to(self.G.device)
      start = 0
      if prompts is not None:
        lcd = prompts['lcd'].flatten(-2).type(batch['lcd'].dtype)
        batch['lcd'][:, :10] = lcd
        batch['proprio'][:, :10] = prompts['proprio']
        start = lcd.shape[1]

      for i in range(start, self.block_size):
        # TODO: check this setting since we have modified things
        logits = self.forward(batch)
        dist = self.dist_head(logits)
        sample = dist.sample()
        batch['lcd'][:, i] = sample[:, i, :self.G.imsize]
        proprio_code = sample[:, i, self.G.imsize:]
        proprio = self.svae.decoder(proprio_code).mean
        batch['proprio'][:, i] = proprio

        if i == self.block_size - 1:
          sample_loss = self.loss(batch)[0]
    batch['lcd'] = batch['lcd'].reshape(n, -1, 1, self.G.lcd_h, self.G.lcd_w)
    return batch, sample_loss.mean().cpu().detach()