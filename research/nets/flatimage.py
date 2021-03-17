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
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead, aggregate, MultiHead, ConvEmbed, ConvBinHead
import utils

class FlatImageTransformer(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.env = env
    self.imsize = self.C.lcd_h * self.C.lcd_w
    self.act_n = env.action_space.shape[0]
    self.block_size = self.C.window

    self.size = self.imsize
    self.gpt_size = self.imsize
    self.dist = self.C.decode
    self.block_size = self.C.window

    self.pos_emb = nn.Parameter(th.zeros(1, self.block_size, C.n_embed))
    self.cond_in = nn.Linear(self.act_n, C.n_embed // 2, bias=False)
    # input embedding stem
    self.embed = nn.Linear(self.gpt_size, C.n_embed // 2, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[Block(self.block_size, C) for _ in range(C.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(C.n_embed)
    if self.C.conv_io:
      self.dist_head = ConvBinHead(C.n_embed, self.gpt_size, C)
      self.custom_embed = ConvEmbed(self.imsize, C.n_embed//2, C)
    else:
      self.dist_head = BinaryHead(C.n_embed, self.gpt_size, C)
    self.optimizer = Adam(self.parameters(), lr=C.lr)
    self.to(C.device)

  def save(self, dir):
    print("SAVED MODEL", dir)
    path = dir / 'flattie.pt'
    th.save(self.state_dict(), path)
    print(path)
  def load(self, path):
    path = path / 'flattie.pt'
    self.load_state_dict(th.load(path))
    print(f'LOADED {path}')


  def forward(self, batch):
    BS, EPL, *HW = batch['lcd'].shape
    lcd = batch['lcd'].reshape(BS, EPL, np.prod(HW))
    acts = batch['acts']
    x = lcd
    BS, T, E = x.shape
    # SHIFT RIGHT (add a padding on the left)
    x = th.cat([th.zeros(BS, 1, E).to(self.C.device), x[:, :-1]], dim=1)
    # forward the GPT model
    if self.C.conv_io:
      x = self.custom_embed(x)
    x = self.embed(x)
    cin = self.cond_in(acts)
    if acts.ndim == 2:
      x = th.cat([x, cin[:, None].repeat_interleave(self.block_size, 1)], -1)
    else:
      x = th.cat([x, cin], -1)
    x += self.pos_emb  # each position maps to a (learnable) vector
    x = self.blocks(x)
    logits = self.ln_f(x)
    return self.dist_head(logits)

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
    dist = self.forward(batch)
    lcd_loss = -dist.log_prob(batch['lcd'].reshape(BS, EPL, np.prod(HW))).mean()
    metrics['loss/lcd'] = lcd_loss
    loss = lcd_loss
    return loss, metrics

  def onestep(self, batch, i):
    dist = self.forward(batch)
    batch['lcd'][:, i + 1] = dist.sample()[:, i]
    return batch

  def sample(self, n, acts=None, prompts=None):
    # TODO: feed act_n
    with th.no_grad():
      if acts is not None:
        n = acts.shape[0]
      batch = {}
      batch['lcd'] = th.zeros(n, self.block_size, self.imsize).to(self.C.device)
      batch['acts'] = acts if acts is not None else (th.rand(n, self.block_size, self.act_n) * 2 - 1).to(self.C.device)
      start = 0
      if prompts is not None:
        lcd = prompts['lcd'].flatten(-2).type(batch['lcd'].dtype)
        batch['lcd'][:, :5] = lcd
        start = lcd.shape[1]

      for i in range(start, self.block_size):
        # TODO: check this setting since we have modified things
        dist = self.forward(batch)
        batch['lcd'][:, i] = dist.sample()[:, i]
        if i == self.block_size - 1:
          sample_loss = self.loss(batch)[0]
    batch['lcd'] = batch['lcd'].reshape(n, -1, 1, self.C.lcd_h, self.C.lcd_w)
    return batch, sample_loss.mean().cpu().detach()

  def evaluate(self, writer, batch, epoch):
    N = self.C.num_envs
    # unpropted
    acts = (th.rand(N, self.C.window, self.env.action_space.shape[0]) * 2 - 1).to(self.C.device)
    sample, sample_loss = self.sample(N, acts=acts)
    lcd = sample['lcd']
    lcd = lcd.cpu().detach().repeat_interleave(4, -1).repeat_interleave(4, -2)[:, 1:]
    #writer.add_video('lcd_samples', utils.force_shape(lcd), epoch, fps=self.C.fps)
    utils.add_video(writer, 'lcd_samples', utils.force_shape(lcd), epoch, fps=self.C.fps)
    # prompted
    if len(self.env.world_def.robots) == 0:  # if we are just dropping the object, always use the same setup
      if 'BoxOrCircle' == self.C.env:
        reset_states = np.c_[np.ones(N), np.zeros(N), np.linspace(-0.8, 0.8, N), 0.5 * np.ones(N)]
      else:
        reset_states = np.c_[np.random.uniform(-1,1,N), np.random.uniform(-1,1,N), np.linspace(-0.8, 0.8, N), 0.5 * np.ones(N)]
    else:
      reset_states = [None] * N
    obses = {key: [[] for ii in range(N)] for key in self.env.observation_space.spaces}
    acts = [[] for ii in range(N)]
    for ii in range(N):
      for key, val in self.env.reset(reset_states[ii]).items():
        obses[key][ii] += [val]
      for _ in range(self.C.window - 1):
        act = self.env.action_space.sample()
        obs = self.env.step(act)[0]
        for key, val in obs.items():
          obses[key][ii] += [val]
        acts[ii] += [act]
      acts[ii] += [np.zeros_like(act)]
    obses = {key: np.array(val) for key, val in obses.items()}
    acts = np.array(acts)
    acts = th.as_tensor(acts, dtype=th.float32).to(self.C.device)
    prompts = {key: th.as_tensor(1.0 * val[:, :5]).to(self.C.device) for key, val in obses.items()}
    prompted_samples, prompt_loss = self.sample(N, acts=acts, prompts=prompts)
    real_lcd = obses['lcd'][:, :, None]
    lcd_psamp = prompted_samples['lcd']
    lcd_psamp = lcd_psamp.cpu().detach().numpy()
    error = (lcd_psamp - real_lcd + 1.0) / 2.0
    blank = np.zeros_like(real_lcd)[..., :1, :]
    out = np.concatenate([real_lcd, blank, lcd_psamp, blank, error], 3)
    out = out.repeat(4, -1).repeat(4, -2)
    #writer.add_video('prompted_lcd', utils.force_shape(out), epoch, fps=self.C.fps)
    utils.add_video(writer, 'prompted_lcd', utils.force_shape(out), epoch, fps=self.C.fps)