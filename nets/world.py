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
    self.block_size = self.C.ep_len
    self.temporal = GPTWorld(size=self.C.lcd_h*self.C.lcd_w, block_size=self.C.ep_len, dist=self.C.decode, cond=self.act_n, C=C)
    #self.temporal = GPTWorld(size=self.C.lcd_h*self.C.lcd_w + 128, block_size=self.C.ep_len, dist=self.C.decode, cond=self.act_n, C=C)
    #self.temporal = GPT(size=self.C.lcd_h*self.C.lcd_w, block_size=self.C.ep_len, dist=self.C.decode, C=C)
    self.state_n = env.observation_space.spaces['state'].shape[0]
    lcd_n = C.lcd_h*C.lcd_w
    self.linear_up = nn.Linear(self.state_n, 128)
    self.to(C.device)

  def forward(self, batch):
    BS, EPL,*HW = batch['lcd'].shape
    lcd = batch['lcd'].reshape(BS, EPL, np.prod(HW))
    state = batch['state']
    acts = batch['acts']
    zstate = self.linear_up(state)
    x = torch.cat([lcd, zstate], -1)
    # should the dist here be independent? prolly not
    bindist, mdndist = self.temporal.forward(lcd, cond=acts)
    #bindist, mdndist = self.temporal.forward(x, cond=acts)
    return bindist, mdndist

  def loss(self, batch):
    bindist, mdndist = self.forward(batch)
    lcd_loss = -bindist.log_prob(batch['lcd'].flatten(-2)).mean()
    state_loss = -mdndist.log_prob(batch['state']).mean()
    return lcd_loss, bindist
    #return lcd_loss+state_loss, bindist

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
          logp = bindist.log_prob(batch['lcd'])

    batch['lcd'] = batch['lcd'].reshape(n, -1, 1, self.C.lcd_h, self.C.lcd_w)
    return batch, logp.mean().item()