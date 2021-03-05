import time
import copy
from sync_vector_env import SyncVectorEnv
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
import torch
import torch as th
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import yaml
from datetime import datetime
import argparse

from boxLCD.utils import A
import utils
import data
from runners.trainer import Trainer
from define_config import env_fn

class EncDecTrainer(Trainer):
  def __init__(self, C):
    super().__init__(C)
    self.optimizer = Adam(self.model.parameters(), lr=C.lr)
    self.scaler = torch.cuda.amp.GradScaler(enabled=C.amp)
    self.C.num_vars = self.num_vars = utils.count_vars(self.model)
    print('num vars', self.num_vars)

  def train_epoch(self, i):
    self.optimizer.zero_grad()
    for batch in self.train_ds:
      batch = {key: val.to(self.C.device) for key, val in batch.items()}
      if self.C.amp:
        with torch.cuda.amp.autocast():
          loss, metrics = self.model.loss(batch)
      else:
          loss, metrics = self.model.loss(batch)
      for key in metrics:
        self.logger[key] += [metrics[key].cpu().detach()]
      self.scaler.scale(loss).backward()
      self.scaler.unscale_(self.optimizer)
      self.logger['grad_norm'] += [torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.C.grad_clip).cpu()]
      self.scaler.step(self.optimizer)
      self.scaler.update()
      self.optimizer.zero_grad()
      self.logger['loss'] += [loss.detach().cpu()]

  def sample(self, i):
    import ipdb; ipdb.set_trace()
    # TODO: prompt to a specific point and sample from there. to compare against ground truth.
    N = self.C.num_envs
    if True:
      acts = (torch.rand(N, self.C.ep_len, self.env.action_space.shape[0])*2 - 1).to(self.C.device)
      sample, sample_loss = self.model.sample(N, cond=acts)
      self.logger['sample_loss'] += [sample_loss]
      if 'image' in self.C.subset:
        lcd = sample['lcd']
        lcd = lcd.cpu().detach().repeat_interleave(4,-1).repeat_interleave(4,-2)[:,1:]
        self.writer.add_video('lcd_samples', utils.force_shape(lcd), i, fps=50)
      if 'state' in self.C.subset:
        state = sample['state'].cpu()
        state_img = []
        for j in range(state.shape[1]):
          obs = self.tvenv.reset(np.arange(self.C.num_envs), state[:,j])
          state_img += [self.tvenv.render(pretty=True)]
        state_img = np.stack(state_img, 1).transpose(0, 1, -1, 2, 3)[...,-self.C.env_size//2:,:]
        self.writer.add_video('state_samples', utils.force_shape(state_img), i, fps=50)

    if True:
      # EVAL
      if len(self.env.env.world_def.robots) == 0:
        reset_states = np.c_[np.ones(N), np.zeros(N), np.linspace(-0.8, 0.8, N), 0.5 * np.ones(N)]
      else:
        reset_states = [None]*N
      obses = {key: [] for key in self.env.observation_space.spaces}
      for key, val in self.tvenv.reset(np.arange(N), reset_states).items():
        obses[key] += [val]
      real_imgs = [self.tvenv.render(pretty=True)]
      acts = []
      for _ in range(self.C.ep_len - 1):
        act = self.tvenv.action_space.sample()
        obs = self.tvenv.step(act)[0]
        for key, val in obs.items(): obses[key] += [val]
        real_imgs += [self.tvenv.render(pretty=True)]
        acts += [act]
      acts += [np.zeros_like(act)]
      obses = {key: np.stack(val, 1) for key, val in obses.items()}
      acts = np.stack(acts, 1)
      acts = torch.as_tensor(acts, dtype=torch.float32).to(self.C.device)
      prompts = {key: torch.as_tensor(1.0*val[:,:5]).to(self.C.device) for key, val in obses.items()}
      prompted_samples, prompt_loss = self.model.sample(N, cond=acts, prompts=prompts)
      self.logger['prompt_sample_loss'] += [prompt_loss]
      # TODO: show a pretty example next to these erro plots
      real_lcd = obses['lcd'][:,:,None]
      if 'image' in self.C.subset:
        lcd_psamp = prompted_samples['lcd']
        lcd_psamp = lcd_psamp.cpu().detach().numpy()

        error = (lcd_psamp - real_lcd + 1.0) / 2.0
        blank = np.zeros_like(real_lcd)[...,:1,:]
        out = np.concatenate([real_lcd, blank, lcd_psamp, blank, error], 3)
        out = out.repeat(4, -1).repeat(4,-2)
        self.writer.add_video('prompted_lcd', utils.force_shape(out), i, fps=50)
      if 'state' in self.C.subset:
        state_psamp = prompted_samples['state'].cpu()
        imgs = []
        for j in range(state_psamp.shape[1]):
          obs = self.big_tvenv.reset(np.arange(self.C.num_envs), state_psamp[:,j])
          if self.C.cheap_render:
            imgs += [obs['lcd'][...,None]]
          else:
            imgs += [self.big_tvenv.render()]
        imgs = np.stack(imgs, 1).transpose(0, 1, -1, 2, 3)[...,-self.C.env_size//2:,:]
        real_imgs = np.stack(real_imgs, 1).transpose(0, 1, -1, 2, 3)[...,-self.C.env_size//2:,:]
        if imgs.dtype == np.uint8:
          error = (real_imgs - imgs + 255) // 2
        else:
          error = (1.0*real_imgs - 1.0*imgs + 1.0) / 2.0

        blank = np.zeros_like(real_imgs)[...,:1,:]
        out = np.concatenate([real_imgs, blank, imgs, blank, error], 3)
        self.logger['prompt_state_img_delta_metric'] = 1e3*((1.0*real_imgs - 1.0*imgs)**2).mean()
        self.writer.add_video('prompted_state', utils.force_shape(out), i, fps=50)

  def test(self, i):
    self.model.eval()
    with torch.no_grad():
      for batch in self.test_ds:
        batch = {key: val.to(self.C.device) for key, val in batch.items()}
        loss, metrics = self.model.loss(batch)
        self.logger['test_loss'] += [loss.mean().detach().cpu()]
    #sample_start = time.time()
    #self.sample(i)
    #self.logger['dt/sample'] = [time.time()-sample_start]
    loss, metrics = self.model.loss(batch, eval=True)
    lcd = batch['lcd'][:8]
    decoded = 1.0*(metrics.pop('decoded')[:8].exp() > 0.5)
    error = (decoded - lcd + 1.0) / 2.0
    stack = th.cat([lcd, decoded, error], -2)
    self.writer.add_image('decode', utils.combine_imgs(stack, 1, 8)[None], i)
    self.logger['num_vars'] = self.num_vars
    self.logger = utils.dump_logger(self.logger, self.writer, i, self.C)
    self.writer.flush()
    self.model.train()

  def run(self):
    for i in itertools.count():
      if not self.C.skip_train:
        self.train_epoch(i)
      self.test(i)

      if (self.C.logdir / 'pause.marker').exists():
        import ipdb; ipdb.set_trace()

      if i % self.C.save_n == 0:
        torch.save(self.model.state_dict(), self.C.logdir / 'weights.pt')

      if i >= self.C.done_n:
        break