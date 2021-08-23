import pickle
from collections import defaultdict
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
from research.nets import net_map

class Evaler:
  def __init__(self, model, env, G):
    super().__init__()
    print('wait dataload')
    self.train_ds, self.test_ds = data.load_ds(G)
    assert self.test_ds.nbarrels == 10, "usually we assume 10 barrels of test data to run this."
    print('dataloaded')
    self.env = env
    sd = th.load(G.weightdir / f'{G.model}.pt')
    mG = sd.pop('G')
    mG.device = G.device
    model = net_map[G.model](env, mG)
    model.load(G.weightdir)
    model.eval()
    model.to(G.device)
    self.model = model
    self.G = G
    self.b = lambda x: {key: val.to(G.device) for key, val in x.items()}
    if G.arbiterdir.name != '':
      arbiter_path = list(G.arbiterdir.glob('*.pt'))
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
    """
    it runs through the test and train datasets 5 times each and averages the stats.
    
    can take quite awhile to run
    """
    self.model.eval()
    self.N = 1e4
    with th.no_grad():
      logger = defaultdict(lambda: [])
      for i in range(5):
        test_logger = self.do_ds(self.test_ds)
        train_logger = self.do_ds(self.train_ds)
        for key in test_logger:
          logger['test:'+key] += [test_logger[key]]
        for key in train_logger:
          logger['train:'+key] += [train_logger[key]]
      final_logger = {}
      for key in logger:
        final_logger[key] = np.mean(logger[key]), np.std(logger[key])
      self.G.logdir.mkdir(parents=True, exist_ok=True)
      with open(self.G.logdir / 'logger.pkl', 'wb') as f:
        pickle.dump(final_logger, f)
      print('wrote pickle', self.G.logdir)
      test = utils.filtdict(final_logger, 'test:', fkey=lambda x:x[5:])
      testu = utils.filtdict(test, 'u:', fkey=lambda x:x[2:])
      testp = utils.filtdict(test, 'p:', fkey=lambda x:x[2:])
      print()
      print('Test Unprompted'+'-'*15)
      for key, val in testu.items(): print(f'{key}: {val[0]}  +/-  {val[1]}')
      print()
      print('Test Prompted'+'-'*15)
      for key, val in testp.items(): print(f'{key}: {val[0]}  +/-  {val[1]}')
      train = utils.filtdict(final_logger, 'train:', fkey=lambda x:x[6:])
      trainu = utils.filtdict(train, 'u:', fkey=lambda x:x[2:])
      trainp = utils.filtdict(train, 'p:', fkey=lambda x:x[2:])
      print()
      print('Train Unprompted'+'-'*15)
      for key, val in trainu.items(): print(f'{key}: {val[0]}  +/-  {val[1]}')
      print()
      print('Train Prompted'+'-'*15)
      for key, val in trainp.items(): print(f'{key}: {val[0]}  +/-  {val[1]}')

  def do_ds(self, ds):
    logger = defaultdict(lambda: [])
    plogger = defaultdict(lambda: [])
    # compute loss on all data
    all_paz = []
    all_upaz = []
    all_taz = []
    for i, test_batch in enumerate(self.test_ds):
      batch = self.b(test_batch)
      upaz, umetrics = self.unprompted(batch)
      for key in umetrics:
        logger['u:'+key] += [umetrics[key].cpu()]
      paz, taz, pmetrics = self.prompted(batch)
      for key in pmetrics:
        logger['p:'+key] += [pmetrics[key].cpu()]
      all_paz += [paz]
      all_upaz += [upaz]
      all_taz += [taz]
      print((i+1) * self.G.bs)
      if (i+1) * self.G.bs >= self.N:
        break
    paz = th.cat(all_paz)
    upaz = th.cat(all_upaz)
    taz = th.cat(all_taz)
    for key, val in self.compute_agged(upaz, taz).items():
      logger['u:'+key] += [val]
    for key, val in self.compute_agged(paz, taz).items():
      logger['p:'+key] += [val]
    for key in logger: logger[key] = np.mean(logger[key])
    return dict(logger)

  def compute_agged(self, paz, taz):
    metrics = {}
    fvd = utils.compute_fid(paz.cpu().numpy(), taz.cpu().numpy())
    metrics['fvd'] = fvd
    precision, recall, f1 = utils.precision_recall_f1(taz[:5000], paz[:5000], k=3)
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
