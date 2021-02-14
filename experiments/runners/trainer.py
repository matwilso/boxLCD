import copy
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

from boxLCD.utils import A
import utils
import data
from runners.runner import Runner
from define_config import env_fn
from nets.state import StateTransformer

class Trainer(Runner):
  def __init__(self, C):
    super().__init__(C)
    C.block_size = C.ep_len
    print('wait dataload')
    self.train_ds, self.test_ds = data.load_ds(C)
    print('dataloaded')
    self.writer = SummaryWriter(C.logdir)
    self.logger = utils.dump_logger({}, self.writer, 0, C)
    self.tvenv = SyncVectorEnv([env_fn(C, 0 + i) for i in range(C.num_envs)], C=C)  # test vector env

    bigC = copy.deepcopy(C)
    bigC.lcd_h *= 4
    bigC.lcd_w *= 4
    self.big_tvenv = SyncVectorEnv([env_fn(bigC, 0 + i) for i in range(C.num_envs)], C=bigC)  # test vector env