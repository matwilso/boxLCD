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
          with Timer(self.logger, 'evaluate'):
            # run the model specific evaluate functtest_timelly draws samples and creates other relevant visualizations.
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
