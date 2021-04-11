import time
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
import utils
from utils import Timer
import data
from define_config import env_fn
import ignite
from rl.async_vector_env import AsyncVectorEnv

class Trainer:
  def __init__(self, model, env, C):
    super().__init__()
    print('wait dataload')
    self.train_ds, self.test_ds = data.load_ds(C)
    print('dataloaded')
    if C.phase == 2:
      C.logdir = C.logdir / 'phase2'
    self.writer = SummaryWriter(C.logdir)
    #self.writer.add_hparams({
    #    'lr': C.lr,
    #    'bs': C.bs,
    #})
    self.logger = utils.dump_logger({}, self.writer, 0, C)
    self.env = env
    self.model = model
    self.num_vars = utils.count_vars(self.model)
    print('num_vars', self.num_vars)
    self.C = C
    self.b = lambda x: {key: val.to(C.device) for key, val in x.items()}
    self.venv = AsyncVectorEnv([env_fn(C) for _ in range(self.C.num_envs)])
    self.ssim = ignite.metrics.SSIM(1.0, device=self.C.device)
    self.psnr = ignite.metrics.PSNR(1.0, device=self.C.device)

  def evaluate(self, batch, itr):
    N = self.C.num_envs
    # UNPROMPTED SAMPLE
    acts = (th.rand(N, self.C.window, self.env.action_space.shape[0]) * 2 - 1).to(self.C.device)
    sample, sample_loss = self.model.sample(N, acts=acts)
    if 'lcd' in sample:
      lcd = sample['lcd']
      lcd = lcd.cpu().detach().repeat_interleave(4, -1).repeat_interleave(4, -2)[:, 1:]
      #writer.add_video('lcd_samples', utils.force_shape(lcd), itr, fps=self.C.fps)
      utils.add_video(self.writer, 'unprompted/lcd', utils.force_shape(lcd), itr, fps=self.C.fps)
    if 'pstate' in sample:
      # TODO: add support for pstate
      import ipdb; ipdb.set_trace()

    # PROMPTED SAMPLE
    prompted_samples, prompt_loss = self.model.sample(N, acts=batch['acts'], prompts=batch, prompt_n=10)
    pred_lcd = prompted_samples['lcd']
    real_lcd = batch['lcd'][:,:,None]
    # metrics 
    self.ssim.update((pred_lcd.flatten(0,1), real_lcd.flatten(0,1)))
    ssim = self.ssim.compute().cpu().detach()
    self.logger['eval/ssim'] += [ssim]
    # TODO: try not flat
    self.psnr.update((pred_lcd.flatten(0,1), real_lcd.flatten(0,1)))
    psnr = self.psnr.compute().cpu().detach()
    self.logger['eval/psnr'] += [psnr]

    if 'pstate' in prompted_samples:
      import ipdb; ipdb.set_trace()
      pstate_samp = prompted_samples['pstate'].cpu().numpy()
      imgs = []
      shape = pstate_samp.shape
      for ii in range(shape[0]):
        col = []
        for jj in range(shape[1]):
          col += [self.env.reset(pstate=pstate_samp[ii, jj])['lcd']]
        imgs += [col]
      pstate_img = 1.0 * np.array(imgs)[:, :, None]
      error = (pstate_img - real_lcd + 1.0) / 2.0
      blank = np.zeros_like(real_lcd)[..., :1, :]
      out = np.concatenate([real_lcd, blank, pstate_img, blank, error], 3)
      out = out.repeat(4, -1).repeat(4, -2)
      utils.add_video(writer, 'prompted_state', utils.force_shape(out), epoch, fps=self.C.fps)


    # visualization
    pred_lcd = pred_lcd[:8].cpu().detach().numpy()
    real_lcd = real_lcd[:8].cpu().detach().numpy()
    error = (pred_lcd - real_lcd + 1.0) / 2.0
    blank = np.zeros_like(real_lcd)[..., :1, :]
    out = np.concatenate([real_lcd, blank, pred_lcd, blank, error], 3)
    out = out.repeat(4, -1).repeat(4, -2)
    #writer.add_video('prompted_lcd', utils.force_shape(out), itr, fps=self.C.fps)
    utils.add_video(self.writer, 'prompted_lcd', utils.force_shape(out), itr, fps=self.C.fps)
    return prompted_samples['lcd']

  def run(self):
    total_time = time.time()
    epoch_time = time.time()
    last_save = time.time()
    train_iter = iter(self.train_ds)
    for itr in itertools.count(1):
      # TRAIN
      with Timer(self.logger, 'sample_batch'):
        train_batch = self.b(next(train_iter))
      with Timer(self.logger, 'train_step'):
        metrics = self.model.train_step(train_batch)
        for key in metrics:
          self.logger[key] += [metrics[key].detach().cpu()]

      if (self.C.logdir / 'pause.marker').exists():
        import ipdb; ipdb.set_trace()

      if itr % self.C.log_n == 0 or self.C.skip_train:
        self.model.eval()
        with th.no_grad():
          with Timer(self.logger, 'test'):
            # compute loss on all data
            for test_batch in self.test_ds:
              metrics = self.model.train_step(self.b(test_batch), dry=True)
              for key in metrics:
                self.logger['test/' + key] += [metrics[key].detach().cpu()]
              break
          with Timer(self.logger, 'evaluate'):
            # run the model specific evaluate functtest_timelly draws samples and creates other relevant visualizations.
            #samples = self.evaluate(self.b(test_batch), itr)
            self.model.evaluate(self.writer, self.b(test_batch), itr)
        self.model.train()

        # LOGGING
        self.logger['dt/total'] = time.time() - total_time
        self.logger['dt/epoch'] = time.time() - epoch_time
        epoch_time = time.time()
        self.logger['num_vars'] = self.num_vars
        self.logger = utils.dump_logger(self.logger, self.writer, itr, self.C)
        self.writer.flush()
        if time.time() - last_save >= 300 or itr % (self.C.log_n * self.C.save_n) == 0:
          self.model.save(self.C.logdir)
          last_save = time.time()
      if itr >= self.C.total_itr:
        break
