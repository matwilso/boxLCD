from sync_vector_env import SyncVectorEnv
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import yaml
from datetime import datetime
import argparse
from envs.box import Box

from utils import A
import utils
from data import data
from envs.box import Box
from runners.trainer import Trainer
from define_config import env_fn
from nets.state import StateTransformer
from nets.pixels import AutoWorldModel

class VideoTrainer(Trainer):
  def __init__(self, C):
    super().__init__(C)
    if C.lcd_render:
       self.model = AutoWorldModel(self.env, C)
    else:
      self.model = StateTransformer(self.env, C)
    self.optimizer = Adam(self.model.parameters(), lr=C.lr)
    self.scaler = torch.cuda.amp.GradScaler(enabled=C.amp)

  def train_epoch(self, i):
    self.optimizer.zero_grad()
    for batch in self.train_ds:
      batch = {key: val.to(self.C.device) for key, val in batch.items()}
      if self.C.amp:
        with torch.cuda.amp.autocast():
          loss, dist = self.model.loss(batch)
      else:
          loss, dist = self.model.loss(batch)
      self.scaler.scale(loss).backward()
      self.scaler.step(self.optimizer)
      self.scaler.update()
      #loss.backward()
      #self.optimizer.step()
      self.optimizer.zero_grad()
      self.logger['loss'] += [loss.detach().cpu()]
      #std = dist.stddev.detach()
      #self.logger['std/mean'] += [std.mean().cpu()]
      #self.logger['std/min'] += [std.min().cpu()]
      #self.logger['std/max'] += [std.max().cpu()]

  def sample(self, i):
    # TODO: prompt to a specific point and sample from there. to compare against ground truth.
    N = self.C.num_envs
    sample, logp = self.model.sample(N)
    self.logger['sample_logp'] = logp
    shape = sample.shape
    sample = torch.roll(sample.view(200, 16*16), -1, -1).view(*shape)
    self.writer.add_video('samples', sample, i, fps=50)

    if False:
      # EVAL
      reset_states = np.c_[np.ones(N), np.zeros(N), np.linspace(-0.8, 0.8, N), 0.5 * np.ones(N)]
      rollout = [self.tvenv.reset(np.arange(N), reset_states)]
      real_imgs = [self.tvenv.render()]
      acts = []
      for _ in range(self.C.ep_len - 1):
        act = self.tvenv.action_space.sample()
        obs = self.tvenv.step(act)[0]
        real_imgs += [self.tvenv.render()]
        rollout += [obs]
        acts += [act]
      rollout = np.stack(rollout, 1)
      acts = np.stack(acts, 1)
      prompts = rollout[:, :5]

      fake_imgs = []
      # TODO: have real env samples side-by-side. probably doing that dreamer error thing
      # TODO: also have many envs in parallel to gen stuff.
      sample, logp = self.model.sample(N, prompts=prompts)
      sample = sample.cpu().numpy()
      self.logger['sample_logp'] = logp
      for j in range(self.C.ep_len - 1):
        obs = self.tvenv.reset(np.arange(N), sample[:, j + 1])
        fake_imgs += [self.tvenv.render()]
      fake_imgs = np.stack(fake_imgs, axis=1).transpose(0, 1, -1, 2, 3)  # / 255.0
      real_imgs = np.stack(real_imgs, axis=1).transpose(0, 1, -1, 2, 3)[:, :self.C.ep_len - 1]  # / 255.0
      error = (fake_imgs - real_imgs + 255) // 2
      #out = error
      out = np.concatenate([real_imgs, fake_imgs, error], 3)
      N, T, C, H, W = out.shape
      out = out.transpose(1, 2, 3, 0, 4).reshape([T, C, H, N * W])[None]
      self.writer.add_video('error', out, i, fps=50)
      #self.writer.add_video('true', real_imgs, i, fps=50)
      #import ipdb; ipdb.set_trace()
      delta = np.abs(rollout[:, :-1] - sample[:, 1:]).mean()
      self.logger['sample_delta'] = [delta]

  def test(self, i):
    self.model.eval()
    total_loss = 0.0
    with torch.no_grad():
      for batch in self.test_ds:
        batch = {key: val.to(self.C.device) for key, val in batch.items()}
        loss, dist = self.model.loss(batch)
        total_loss += loss * batch['acts'].shape[0]
      avg_loss = total_loss / len(self.test_ds.dataset)
    self.logger['test/bits_per_dim'] = avg_loss.item() / np.log(2)
    self.sample(i)
    self.logger = utils.dump_logger(self.logger, self.writer, i, self.C)
    self.writer.flush()
    self.model.train()

  def run(self):
    for i in itertools.count():
      if not self.C.skip_train:
        self.train_epoch(i)
      self.test(i)
