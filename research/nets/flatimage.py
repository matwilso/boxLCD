import sys
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead, aggregate, MultiHead, ConvEmbed

class FlatImageTransformer(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.imsize = self.C.lcd_h*self.C.lcd_w
    self.act_n = env.action_space.shape[0]
    if self.C.full_state:
      self.state_n = env.observation_space.spaces['full_state'].shape[0]
    else:
      self.state_n = env.observation_space.spaces[''pstate'].shape[0]
    self.block_size = self.C.ep_len
    C.state_n = self.C.state_n = self.state_n

    self.size = 0
    self.gpt_size = 0
    if 'image' in self.C.subset:
      self.size += self.imsize
      self.gpt_size += self.imsize
    if 'pstate' in self.C.subset:
      self.size += self.state_n
      self.gpt_size += 128

    self.dist = self.C.decode
    self.block_size = self.C.ep_len

    self.linear_up = nn.Linear(self.state_n, 128)
    self.to(C.device)

    self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, C.n_embed))
    self.cond_in = nn.Linear(self.act_n, C.n_embed//2, bias=False)
    # input embedding stem
    self.embed = nn.Linear(self.gpt_size, C.n_embed//2, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[Block(self.block_size, C) for _ in range(C.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(C.n_embed)
    if self.dist == 'gauss':
      self.dist_head = GaussHead(self.gpt_size, C)
    elif self.dist == 'mdn':
      self.dist_head = MDNHead(self.gpt_size, C)
    elif self.dist == 'binary':
      self.dist_head = BinaryHead(C.n_embed, self.gpt_size, C)
    elif self.dist == 'multi':
      if self.C.conv_io:
        self.dist_head = MultiHead(C.n_embed, 256, 128, C)
      else:
        self.dist_head = MultiHead(C.n_embed, C.state_n+self.imsize, self.imsize, C)
    if self.C.conv_io:
      self.custom_embed = ConvEmbed(self.imsize, C.n_embed//2, C)
    self.to(C.device)


  def forward(self, batch):
    BS, EPL,*HW = batch['lcd'].shape
    lcd = batch['lcd'].reshape(BS, EPL, np.prod(HW))
    state = batch[''pstate']
    acts = batch['acts']
    if 'pstate' == self.C.subset:
      zstate = self.linear_up(state)
      x = zstate
    if 'image' == self.C.subset:
      x = lcd
    if 'image' in self.C.subset and 'pstate' in self.C.subset:
      zstate = self.linear_up(state)
      x = torch.cat([lcd, zstate], -1)
    BS, T, E = x.shape
    # SHIFT RIGHT (add a padding on the left)
    x = torch.cat([torch.zeros(BS, 1, E).to(self.C.device), x[:, :-1]], dim=1)
    # forward the GPT model
    if self.C.conv_io:
      x = self.custom_embed(x)
    x = self.embed(x)
    cin = self.cond_in(acts)
    if acts.ndim == 2:
      x = torch.cat([x, cin[:,None].repeat_interleave(self.block_size, 1)], -1)
    else:
      x = torch.cat([x, cin], -1)
    x += self.pos_emb # each position maps to a (learnable) vector
    x = self.blocks(x)
    logits = self.ln_f(x)
    return self.dist_head(logits)

  def loss(self, batch):
    BS, EPL,*HW = batch['lcd'].shape
    metrics = {}
    #noise_batch = {key: val for key,val in batch.items()}
    #noise_batch['pstate'] += 1e-2*torch.randn(batch['pstate'].shape).to(batch['pstate'].device)

    bindist, statedist = self.forward(batch)
    lcd_loss = torch.zeros(1, device=self.C.device)
    state_loss = torch.zeros(1, device=self.C.device)
    if 'image' in self.C.subset:
      lcd_loss = -bindist.log_prob(batch['lcd'].reshape(BS, EPL, np.prod(HW))).mean()
      metrics['loss/lcd'] = lcd_loss
    if 'pstate' in self.C.subset:
      state_loss = -statedist.log_prob(batch['pstate']).mean()
      metrics['loss/state'] = state_loss
    loss = lcd_loss + 1e-4*state_loss
    return loss, metrics

  def onestep(self, batch, i):
    bindist, statedist = self.forward(batch)
    if 'image' in self.C.subset:
      batch['lcd'][:, i+1] = bindist.sample()[:,i]
    return batch

  def sample(self, n, cond=None, prompts=None):
    # TODO: feed act_n
    with torch.no_grad():
      if cond is not None:
        n = cond.shape[0]
      batch = {}
      batch['lcd'] = torch.zeros(n, self.block_size, self.imsize).to(self.C.device)
      batch['pstate'] = torch.zeros(n, self.block_size, self.state_n).to(self.C.device)
      batch['acts'] = cond if cond is not None else (torch.rand(n, self.block_size, self.act_n)*2 - 1).to(self.C.device)
      start = 0
      if prompts is not None:
        lcd = prompts['lcd'].flatten(-2).type(batch['lcd'].dtype)
        batch['lcd'][:, :5] = lcd
        batch['pstate'][:, :5] = prompts['pstate']
        start = lcd.shape[1]

      for i in range(start, self.block_size):
        # TODO: check this setting since we have modified things
        bindist, statedist = self.forward(batch)
        if 'image' in self.C.subset:
         batch['lcd'][:, i] = bindist.sample()[:,i]
        if 'pstate' in self.C.subset:
          batch['pstate'][:, i] = statedist.mean[:,i]
          #batch['pstate'][:, i+1] = statedist.sample()[:,i]

        if i == self.block_size-1:
          sample_loss = self.loss(batch)[0]

    batch['lcd'] = batch['lcd'].reshape(n, -1, 1, self.C.lcd_h, self.C.lcd_w)
    return batch, sample_loss.mean().cpu().detach()