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
from wrappers import AsyncVectorEnv

class Trainer:
  def __init__(self, model, env, G):
    super().__init__()
    print('wait dataload')
    self.train_ds, self.test_ds = data.load_ds(G)
    print('dataloaded')
    if G.phase == 2:
      G.logdir = G.logdir / 'phase2'
    self.writer = SummaryWriter(G.logdir)
    #self.writer.add_hparams({
    #    'lr': G.lr,
    #    'bs': G.bs,
    #})
    self.logger = utils.dump_logger({}, self.writer, 0, G)
    self.env = env
    self.model = model
    self.num_vars = utils.count_vars(self.model)
    print('num_vars', self.num_vars)
    self.G = G
    self.b = lambda x: {key: val.to(G.device) for key, val in x.items()}
    self.venv = AsyncVectorEnv([env_fn(G) for _ in range(self.G.num_envs)])
    self.model.save(self.G.logdir)

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

      if (self.G.logdir / 'pause.marker').exists():
        import ipdb; ipdb.set_trace()

      if itr % self.G.log_n == 0 or self.G.skip_train:
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
            self.model.evaluate(self.writer, self.b(test_batch), itr)
        self.model.train()

        # LOGGING
        self.logger['dt/total'] = time.time() - total_time
        self.logger['dt/epoch'] = time.time() - epoch_time
        epoch_time = time.time()
        self.logger['num_vars'] = self.num_vars
        self.logger = utils.dump_logger(self.logger, self.writer, itr, self.G)
        self.writer.flush()
        if time.time() - last_save >= 300 or itr % (self.G.log_n * self.G.save_n) == 0:
          self.model.save(self.G.logdir)
          last_save = time.time()
      if itr >= self.G.total_itr:
        break
