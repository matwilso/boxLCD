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
from wrappers import AsyncVectorEnv
from jax.tree_util import tree_multimap, tree_map

class Fiddler:
  def __init__(self, model, env, G):
    super().__init__()
    self.env = env
    self.model = model
    self.G = G

  def run(self):
    all_obses = []
    for i in range(5):
      obs = self.env.reset()
      cache = np.array(obs['full_state'][self.env.idxs[:1]])
      obses = [obs]
      for j in range(100):
        #_o = self.env.reset()
        #_o['full_state'][self.env.idxs] = cache
        #obs = self.env.reset(full_state=_o['full_state'])

        obs['full_state'][self.env.idxs[:1]] = np.random.uniform(-1,1)
        obs = self.env.reset(full_state=obs['full_state'])
        self.env.render(mode='human')
        obses += [obs]
      #all_obses += [obses]
      print(i)
      obses = tree_multimap(lambda x, *y: np.stack([x, *y]), obses[0], *obses[1:])
      #obses = tree_map(lambda x: np.stack([*x], 0), obses)
      obses = tree_map(lambda v: th.as_tensor(v, dtype=th.float32).to(self.G.device), obses)
      out = self.model.encode(obses)
      print(out.mean.var(0).topk(10)[1])