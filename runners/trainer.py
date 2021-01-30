from sync_vector_env import SyncVectorEnv
from nets.transformer import Transformer
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
from flags import flags, args_type
from envs.box import Box

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from utils import A
import utils
from data import data
from envs.box import Box

def env_fn(F, seed=None):
  def _make():
    env = Box(F)
    env.seed(seed)
    return env
  return _make

class Trainer:
  def __init__(self, F):
    self.env = Box(F)
    self.F = F
    print('wait dataload')
    print('dataloaded')
    F.block_size = 200
    self.model = Transformer(self.env, F)
    self.optimizer = Adam(self.model.parameters(), lr=F.lr)
    self.writer = SummaryWriter(F.logdir)
    self.logger = utils.dump_logger({}, self.writer, 0, F)
    self.tvenv = SyncVectorEnv([env_fn(F, 0 + i) for i in range(F.num_envs)], F=F)  # test vector env
    self.train_ds, self.test_ds = data.load_ds(F)

  def train_epoch(self, i):
    for batch in self.train_ds:
      batch = {key: val.to(self.F.device) for key, val in batch.items()}
      self.optimizer.zero_grad()
      loss = self.model.nll(batch)
      loss.backward()
      self.optimizer.step()
      self.logger['loss'] += [loss.detach().cpu()]

  def sample(self, i):
    # TODO: make sure we are sampling correctly
    # TODO: seed it to a specific starting point.
    # EVAL
    N = self.F.num_envs
    reset_states = np.c_[np.ones(N), np.zeros(N), np.linspace(-0.8, 0.8, N), 0.5 * np.ones(N)]
    rollout = [self.tvenv.reset(np.arange(N), reset_states)]
    real_imgs = [self.tvenv.render()]
    acts = []
    for _ in range(199):
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
    for j in range(199):
      obs = self.tvenv.reset(np.arange(N), sample[:, j + 1])
      fake_imgs += [self.tvenv.render()]
    fake_imgs = np.stack(fake_imgs, axis=1).transpose(0, 1, -1, 2, 3)# / 255.0
    real_imgs = np.stack(real_imgs, axis=1).transpose(0, 1, -1, 2, 3)[:,:199]# / 255.0
    error = (fake_imgs - real_imgs + 255) // 2
    out = error
    #out = np.concatenate([real_imgs, fake_imgs, error], 3)
    N, T, C, H, W = out.shape
    out = out.transpose(1,2,3,0,4).reshape([T, C, H, N*W])[None]
    self.writer.add_video('error', out, i, fps=50)
    #self.writer.add_video('true', real_imgs, i, fps=50)

  def test(self, i):
    self.model.eval()
    total_loss = 0.0
    with torch.no_grad():
      for batch in self.test_ds:
        batch = {key: val.to(self.F.device) for key, val in batch.items()}
        loss = self.model.nll(batch)
        total_loss += loss * batch['o'].shape[0]
      avg_loss = total_loss / len(self.test_ds.dataset)
    self.logger['test/bits_per_dim'] = avg_loss.item() / np.log(2)
    self.sample(i)
    self.logger = utils.dump_logger(self.logger, self.writer, i, self.F)
    self.writer.flush()
    self.model.train()

  def run(self):
    for i in itertools.count():
      self.train_epoch(i)
      self.test(i)