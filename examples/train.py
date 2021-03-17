import time
from collections import defaultdict
import pathlib
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from boxLCD import envs, env_map
from boxLCD.utils import A, AttrDict, args_type
import copy
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
import torch as th
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import yaml
from datetime import datetime
import argparse

from boxLCD.utils import A
from model import GPT
import utils
from utils import parse_args

class Trainer:
  def __init__(self, C):
    self.C = C
    self.env = env_map[C.env](C)
    self.act_dim = self.env.action_space.shape[0]
    self.C.lcd_w = int(self.C.lcd_base * self.C.wh_ratio)
    self.C.lcd_h = self.C.lcd_base
    self.model = GPT(self.act_dim, C)
    self.optimizer = Adam(self.model.parameters(), lr=C.lr)
    self.C.num_vars = self.num_vars = utils.count_vars(self.model)
    print('num vars', self.num_vars)
    self.train_ds, self.test_ds = utils.load_ds(C)
    self.writer = SummaryWriter(C.logdir)
    self.logger = utils.dump_logger({}, self.writer, 0, C)

  def train_epoch(self, i):
    """run a single epoch of training over the data"""
    self.optimizer.zero_grad()
    for batch in self.train_ds:
      batch = {key: val.to(self.C.device) for key, val in batch.items()}
      self.optimizer.zero_grad()
      loss = self.model.loss(batch)
      loss.backward()
      self.optimizer.step()
      self.logger['loss'] += [loss.detach().cpu()]

  def sample(self, i):
    # TODO: prompt to a specific point and sample from there. to compare against ground truth.
    N = 5
    acts = (th.rand(N, self.C.ep_len, self.env.action_space.shape[0]) * 2 - 1).to(self.C.device)
    sample, sample_loss = self.model.sample(N, acts=acts)
    self.logger['sample_loss'] += [sample_loss]
    lcd = sample['lcd']
    lcd = lcd.cpu().detach().repeat_interleave(4, -1).repeat_interleave(4, -2)[:, 1:]
    self.writer.add_video('lcd_samples', utils.force_shape(lcd), i, fps=self.C.fps)
    # EVAL
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
      for _ in range(self.C.ep_len - 1):
        act = self.env.action_space.sample()
        obs = self.env.step(act)[0]
        for key, val in obs.items():
          obses[key][ii] += [val]
        acts[ii] += [act]
      acts[ii] += [np.zeros_like(act)]
    obses = {key: np.array(val) for key, val in obses.items()}
    acts = np.array(acts)
    acts = th.as_tensor(acts, dtype=th.float32).to(self.C.device)
    prompts = {key: th.as_tensor(1.0 * val[:, :10]).to(self.C.device) for key, val in obses.items()}
    prompted_samples, prompt_loss = self.model.sample(N, acts=acts, prompts=prompts)
    self.logger['prompt_sample_loss'] += [prompt_loss]
    real_lcd = obses['lcd'][:, :, None]
    lcd_psamp = prompted_samples['lcd']
    lcd_psamp = lcd_psamp.cpu().detach().numpy()
    error = (lcd_psamp - real_lcd + 1.0) / 2.0
    blank = np.zeros_like(real_lcd)[..., :1, :]
    out = np.concatenate([real_lcd, blank, lcd_psamp, blank, error], 3)
    out = out.repeat(4, -1).repeat(4, -2)
    self.writer.add_video('prompted_lcd', utils.force_shape(out), i, fps=self.C.fps)

  def test(self, i):
    self.model.eval()
    with th.no_grad():
      for batch in self.test_ds:
        batch = {key: val.to(self.C.device) for key, val in batch.items()}
        loss = self.model.loss(batch)
        self.logger['test_loss'] += [loss.mean().detach().cpu()]
    sample_start = time.time()
    if i % 10 == 0:
      self.sample(i)
    self.logger['dt/sample'] = [time.time() - sample_start]
    self.logger['num_vars'] = self.num_vars
    self.logger = utils.dump_logger(self.logger, self.writer, i, self.C)
    self.writer.flush()
    self.model.train()

  def run(self):
    for i in itertools.count():
      self.train_epoch(i)
      self.test(i)
      if i >= self.C.num_epochs:
        break

if __name__ == '__main__':
  C = parse_args()
  trainer = Trainer(C)
  trainer.run()