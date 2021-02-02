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
from nets.pixel_vae import PixelVAE

class ImageTrainer(Trainer):
  def __init__(self, C):
    super().__init__(C)
    self.model = PixelVAE(self.env, C)
    self.optimizer = Adam(self.model.parameters(), lr=C.lr)

  def train_epoch(self, i):
    ct = 0
    for batch in self.train_ds:
      batch = {key: val.to(self.C.device) for key, val in batch.items()}
      self.optimizer.zero_grad()
      loss, dist = self.model.loss(batch)
      loss.backward()
      self.optimizer.step()
      self.logger['loss'] += [loss.detach().cpu()]
      #std = dist.stddev.detach()
      #self.logger['std/mean'] += [std.mean().cpu()]
      #self.logger['std/min'] += [std.min().cpu()]
      #self.logger['std/max'] += [std.max().cpu()]

  def sample(self, i):
    # TODO: prompt to a specific point and sample from there. to compare against ground truth.
    N = self.C.num_envs
    sample, logp = self.model.sample(N)
    self.logger['sample_nlogp'] = -logp
    self.writer.add_video('samples', sample, i, fps=50)
    #error = (fake_imgs - real_imgs + 255) // 2
    #out = np.concatenate([real_imgs, fake_imgs, error], 3)
    #N, T, C, H, W = out.shape
    #out = out.transpose(1, 2, 3, 0, 4).reshape([T, C, H, N * W])[None]
    #self.writer.add_video('error', out, i, fps=50)

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
    real = batch['lcd'].cpu().detach().reshape(self.C.bs, 1, 16, 16)[:8]
    sample = dist.sample().cpu().detach().reshape(self.C.bs, 1, 16, 16)[:8]
    error = (sample - real + 1.0) / 2.0
    out = np.concatenate([real, sample, error], 2)
    self.writer.add_images('samples', out, i)

    #self.sample(i)

    self.logger = utils.dump_logger(self.logger, self.writer, i, self.C)
    self.writer.flush()
    self.model.train()

  def run(self):
    for i in itertools.count():
      if not self.C.skip_train:
        self.train_epoch(i)
      self.test(i)