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
from research.nets.common import GaussHead, MDNHead, CausalSelfAttention, TransformerBlock, BinaryHead, aggregate, MultiHead, ConvEmbed, ConvBinHead
from research import utils
import ignite
from ._base import VideoModel

class FIT(VideoModel):
  """FlatImageToken"""
  def __init__(self, env, G):
    super().__init__(env, G)
    self.imsize = self.G.lcd_h * self.G.lcd_w
    self.act_n = env.action_space.shape[0]

    self.size = self.imsize
    self.gpt_size = self.imsize
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
    if self.G.conv_io:
      self.dist_head = ConvBinHead(G.n_embed, self.gpt_size, G)
      self.custom_embed = ConvEmbed(self.imsize, G.n_embed // 2, G)
    else:
      self.dist_head = BinaryHead(G.n_embed, self.gpt_size, G)
    self.optimizer = Adam(self.parameters(), lr=G.lr)
    self.to(G.device)

  def forward(self, batch):
    BS, EPL, *HW = batch['lcd'].shape
    lcd = batch['lcd'].reshape(BS, EPL, np.prod(HW))
    action = batch['action']
    x = lcd
    BS, T, E = x.shape
    # SHIFT RIGHT (add a padding on the left)
    x = th.cat([th.zeros(BS, 1, E).to(self.G.device), x[:, :-1]], dim=1)
    action = th.cat([th.zeros(BS, 1, action.shape[-1]).to(self.G.device), action[:,:-1]], dim=1)
    # forward the GPT model
    if self.G.conv_io:
      x = self.custom_embed(x)
    x = self.embed(x)
    cin = self.cond_in(action)
    if action.ndim == 2:
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
    lcd_loss = -dist.log_prob(batch['lcd'].reshape(BS, EPL, np.prod(HW))).mean()
    metrics['loss/lcd'] = lcd_loss
    loss = lcd_loss
    return loss, metrics

  def onestep(self, batch, i, temp=1.0):
    logits = self.forward(batch)
    dist = self.dist_head(logits/temp)
    batch['lcd'][:, i] = dist.sample()[:, i]
    return batch

  def sample(self, n, action=None, prompts=None, prompt_n=10):
    # TODO: feed act_n
    with th.no_grad():
      if action is not None:
        n = action.shape[0]
      batch = {}
      batch['lcd'] = th.zeros(n, self.block_size, self.imsize).to(self.G.device)
      batch['action'] = action if action is not None else (th.rand(n, self.block_size, self.act_n) * 2 - 1).to(self.G.device)
      start = 0
      if prompts is not None:
        lcd = prompts['lcd'].flatten(-2).type(batch['lcd'].dtype)
        batch['lcd'][:, :prompt_n] = lcd[:, :prompt_n]
        start = prompt_n

      for i in range(start, self.block_size):
        # TODO: check this setting since we have modified things
        logits = self.forward(batch)
        dist = self.dist_head(logits)
        batch['lcd'][:, i] = dist.sample()[:, i]
    batch['lcd'] = batch['lcd'].reshape(n, -1, 1, self.G.lcd_h, self.G.lcd_w)
    batch['proprio'] = th.zeros([*batch['lcd'].shape[:2], self.env.observation_space['proprio'].shape[0]]).to(batch['lcd'].device)
    return batch