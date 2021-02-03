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
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead, aggregate
from nets.gpt import GPT
from nets.gpt_world import GPTWorld

class AutoWorld(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.imsize = self.C.lcd_h*self.C.lcd_w
    self.act_n = env.action_space.shape[0]
    self.state_n = env.observation_space.spaces['state'].shape[0]
    self.block_size = self.C.ep_len

    self.size = 0
    self.gpt_size = 0
    if 'image' in self.C.subset:
      self.size += self.imsize
      self.gpt_size += self.imsize
    if 'state' in self.C.subset:
      self.size += self.state_n
      self.gpt_size += 128
    self.temporal = GPTWorld(size=self.gpt_size, block_size=self.C.ep_len, dist=self.C.decode, cond=self.act_n, C=C)
    self.linear_up = nn.Linear(self.state_n, 128)
    self.to(C.device)

  def forward(self, batch):
    BS, EPL,*HW = batch['lcd'].shape
    lcd = batch['lcd'].reshape(BS, EPL, np.prod(HW))
    state = batch['state']
    acts = batch['acts']
    if 'state' == self.C.subset:
      zstate = self.linear_up(state)
      x = zstate
    if 'image' == self.C.subset:
      x = lcd
    if 'image' in self.C.subset and 'state' in self.C.subset:
      zstate = self.linear_up(state)
      x = torch.cat([lcd, zstate], -1)

    bindist, mdndist = self.temporal.forward(x, cond=acts)
    return bindist, mdndist

  def loss(self, batch):
    BS, EPL,*HW = batch['lcd'].shape
    bindist, mdndist = self.forward(batch)
    lcd_loss = -bindist.log_prob(batch['lcd'].reshape(BS, EPL, np.prod(HW))).mean()
    state_loss = -mdndist.log_prob(batch['state']).mean()
    loss = 0.0
    if 'image' in self.C.subset:
      loss += lcd_loss
    if 'state' in self.C.subset:
      loss += state_loss
    return loss, {'loss/state': state_loss, 'loss/image': lcd_loss}

  def sample(self, n, cond=None, prompts=None):
    # TODO: feed act_n
    with torch.no_grad():
      if cond is not None:
        n = cond.shape[0]
      batch = {}
      batch['lcd'] = torch.zeros(n, self.block_size, self.imsize).to(self.C.device)
      batch['state'] = torch.zeros(n, self.block_size, self.state_n).to(self.C.device)
      batch['acts'] = cond if cond is not None else (torch.rand(n, self.block_size, self.act_n)*2 - 1).to(self.C.device)
      start = 0
      if prompts is not None:
        lcd = prompts['lcd'].flatten(-2).type(batch['lcd'].dtype)
        batch['lcd'][:, :5] = lcd
        batch['state'][:, :5] = prompts['state']
        start = lcd.shape[1]-1

      for i in range(start, self.block_size-1):
        # TODO: check this setting since we have modified things
        bindist, mdndist = self.forward(batch)
        batch['lcd'][:, i+1] = bindist.sample()[:,i]
        batch['state'][:, i+1] = mdndist.sample()[:,i]

        if i == self.block_size-2:
          sample_loss = self.loss(batch)[0]

    batch['lcd'] = batch['lcd'].reshape(n, -1, 1, self.C.lcd_h, self.C.lcd_w)
    return batch, sample_loss.mean().cpu().detach()