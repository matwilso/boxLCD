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
from flags import flags, args_type
from envs.box import Box

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from utils import A
import utils

class RolloutDataset(Dataset):
  def __init__(self, npzfile, train=True, F=None):
    data = np.load(npzfile)
    obses = data['obses']
    acts = data['acts']
    cut = int(len(obses) * 0.8)
    if train:
      self.obses = obses[:cut]
      self.acts = acts[:cut]
    else:
      self.obses = obses[cut:]
      self.acts = acts[cut:]

  def __len__(self):
    return len(self.obses)

  def __getitem__(self, idx):
    batch = {'o': self.obses[idx], 'a': self.acts[idx]}
    return {key: torch.as_tensor(val, dtype=torch.float32) for key, val in batch.items()}

def load_ds(F):
  from torchvision import transforms
  train_dset = RolloutDataset('test.npz', train=True, F=F)
  test_dset = RolloutDataset('test.npz', train=False, F=F)
  train_loader = DataLoader(train_dset, batch_size=F.bs, shuffle=True, pin_memory=True, num_workers=2)
  test_loader = DataLoader(test_dset, batch_size=F.bs, shuffle=True, pin_memory=True, num_workers=2)
  return train_loader, test_loader
