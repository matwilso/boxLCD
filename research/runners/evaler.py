import ignite
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
from torch import nn

from boxLCD.utils import A
from research import utils, data
from research.utils import Timer
from research.define_config import env_fn
from research.wrappers import AsyncVectorEnv
from jax.tree_util import tree_map

class Evaler:
  def __init__(self, model, env, G):
    super().__init__()
    print('wait dataload')
    self.train_ds, self.test_ds = data.load_ds(G)
    print('dataloaded')
    self.env = env
    self.model = model
    self.G = G
    self.writer = SummaryWriter(G.logdir)
    self.b = lambda x: {key: val.to(G.device) for key, val in x.items()}
    arbiter_path = list(G.arbiterdir.glob('*.pt'))
    if len(arbiter_path) > 0:
      arbiter_path = arbiter_path[0]
      self.arbiter = th.jit.load(str(arbiter_path))
      with (arbiter_path.parent / 'hps.yaml').open('r') as f:
        arbiterG = yaml.load(f, Loader=yaml.Loader)
      self.arbiter.G = arbiterG
      self.arbiter.eval()
      print('LOADED ARBITER', arbiter_path)
    else:
      self.arbiter = None
    def chop(x):
      T = x.shape[1]
      _chop = T % self.arbiter.G.window
      if _chop != 0:
        x = x[:, :-_chop]
      return x.reshape([-1, self.arbiter.G.window, *x.shape[2:]])
    self.chop = chop
    self.ssim = ignite.metrics.SSIM(1.0, device=self.G.device)
    self.psnr = ignite.metrics.PSNR(1.0, device=self.G.device)
    self.cossim = nn.CosineSimilarity(dim=-1)
    self.model.load(self.G.weightdir)

  def run(self):
    self.model.eval()
    ulogger = utils.dump_logger({}, self.writer, 0, self.G)
    plogger = utils.dump_logger({}, self.writer, 0, self.G)
    with th.no_grad():
        # compute loss on all data
      all_paz = []
      all_upaz = []
      all_taz = []
      for i, test_batch in enumerate(self.test_ds):
        batch = self.b(test_batch)
        upaz, umetrics = self.unprompted(batch)
        for key in umetrics:
          ulogger[key] += [umetrics[key].cpu()]
        paz, taz, pmetrics = self.prompted(batch)
        for key in pmetrics:
          plogger[key] += [pmetrics[key].cpu()]
        all_paz += [paz]
        all_upaz += [upaz]
        all_taz += [taz]
        print(i*self.G.bs)
      paz = th.cat(all_paz)
      upaz = th.cat(all_upaz)
      taz = th.cat(all_taz)
      pm = self.compute_agged(paz, taz)
      um = self.compute_agged(upaz, taz)
      for key in plogger:
        plogger[key] = np.mean(plogger[key])
      for key in ulogger:
        ulogger[key] = np.mean(ulogger[key])
      print()
      print('UNPROMPTED', um)
      print()
      #print(ulogger)
      print('PROMPTED', pm)
      print()
      print(plogger)
      import ipdb; ipdb.set_trace()

  def compute_agged(self, paz, taz):
    metrics = {}
    fvd = utils.compute_fid(paz.cpu().numpy(), taz.cpu().numpy())
    metrics['fvd'] = fvd
    precision, recall, f1 = utils.precision_recall_f1(taz, paz, k=3)
    metrics['precision'] = precision.cpu()
    metrics['recall'] = recall.cpu()
    metrics['f1'] = f1.cpu()
    return metrics


  def unprompted(self, batch):
    # take sample of same size as batch
    n = batch['lcd'].shape[0]
    action = (th.rand(n, self.G.window, self.env.action_space.shape[0]) * 2 - 1).to(self.G.device)
    sample = self.model.sample(n, action)
    # cut off the burnin
    burned = tree_map(lambda x: x[:, self.G.prompt_n:], sample)
    burned['lcd'] = burned['lcd'][:, :, 0]
    swindow = tree_map(self.chop, burned)
    sact = action[:, self.G.prompt_n:]
    sact = self.chop(sact)[:, :-1]
    # run through arbiter
    paz, paa = self.arbiter.forward(swindow)
    action_log_mse = ((sact - paa)**2).mean().log()
    return paz.cpu(), {'action_log_mse': action_log_mse}

  def prompted(self, batch):
    # take sample with batch actions and batch prompt
    n = batch['lcd'].shape[0]
    sample = self.model.sample(n, action=batch['action'], prompts=batch, prompt_n=self.G.prompt_n)
    metrics = {}
    if 'lcd' in sample:
      pred_lcd = sample['lcd'][:, self.G.prompt_n:]
      true_lcd = batch['lcd'][:, :, None][:, self.G.prompt_n:]
      # run basic metrics
      self.ssim.update((pred_lcd.flatten(0, 1), true_lcd.flatten(0, 1)))
      ssim = self.ssim.compute().cpu().detach()
      metrics['ssim'] = ssim
      self.psnr.update((pred_lcd.flatten(0, 1), true_lcd.flatten(0, 1)))
      psnr = self.psnr.compute().cpu().detach()
      metrics['psnr'] = psnr

    if 'proprio' in sample:
      pred_proprio = sample['proprio']
      true_proprio = batch['proprio']
      metrics['proprio_log_mse'] = ((true_proprio[:, self.G.prompt_n:] - pred_proprio[:, self.G.prompt_n:])**2).mean().log().cpu()

    t_burned = tree_map(lambda x: x[:, self.G.prompt_n:], batch)
    s_burned = tree_map(lambda x: x[:, self.G.prompt_n:], sample)
    s_burned['lcd'] = s_burned['lcd'][:, :, 0]

    s_window = tree_map(self.chop, s_burned)
    t_window = tree_map(self.chop, t_burned)
    tact = batch['action'][:, self.G.prompt_n:]
    tact = self.chop(tact)[:, :-1]

    paz, paa = self.arbiter.forward(s_window)
    taz, taa = self.arbiter.forward(t_window)
    metrics['action_log_mse'] = ((tact - paa)**2).mean().log()
    metrics['true_action_log_mse'] = ((tact - taa)**2).mean().log()
    metrics['cosdist'] = 1 - self.cossim(paz, taz).mean().cpu()
    return paz.cpu(), taz.cpu(), metrics
