import time
import copy
from sync_vector_env import SyncVectorEnv
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
import data
from define_config import env_fn


class Trainer:
  def __init__(self, model, env, C):
    super().__init__()
    print('wait dataload')
    self.train_ds, self.test_ds = data.load_ds(C)
    print('dataloaded')
    self.writer = SummaryWriter(C.logdir)
    self.logger = utils.dump_logger({}, self.writer, 0, C)
    self.tvenv = SyncVectorEnv([env_fn(C, 0 + i) for i in range(C.num_envs)], C=C)  # test vector env
    self.env = env
    self.model = model
    self.num_vars = utils.count_vars(self.model)
    self.C = C
    self.b = lambda x: {key: val.to(C.device) for key, val in x.items()}

  def run(self):
    for epoch in itertools.count():
      # TRAIN
      if not self.C.skip_train:
        train_time = time.time()
        for batch in self.train_ds:
          metrics = self.model.train_step(self.b(batch))
        self.logger['dt/train'] = time.time() - train_time
        for key in metrics: self.logger[key] += [metrics[key].detach().cpu()]

      if (self.C.logdir / 'pause.marker').exists():
        import ipdb; ipdb.set_trace()

      # TEST
      self.model.eval()
      with th.no_grad():
        test_batch = self.b(next(iter(self.test_ds)))
        # run the model specific evaluate function. usually draws samples and creates other relevant visualizations.
        eval_time = time.time()
        self.model.evaluate(self.writer, test_batch, epoch)
        self.logger['dt/evaluate'] = time.time() - eval_time
      self.model.train()

      # LOGGING
      self.logger['num_vars'] = self.num_vars
      self.logger = utils.dump_logger(self.logger, self.writer, epoch, self.C)
      self.writer.flush()
      if epoch % self.C.save_n == 0:
        path = self.C.logdir / 'model.pt'
        print("SAVED MODEL", path)
        th.save(self.model.state_dict(), path)
      if epoch >= self.C.num_epochs:
        break


