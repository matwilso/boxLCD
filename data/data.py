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
from envs.box import Box

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from utils import A
import utils

class RolloutDataset(Dataset):
  def __init__(self, npzfile, train=True, C=None):
    data = np.load(npzfile, allow_pickle=True)
    self.bufs = {key: torch.as_tensor(data[key]) for key in data.keys()}
    if C.data_mode == 'image':
      self.bufs = {key: val.flatten(0, 1) for key, val in self.bufs.items()}
    cut = int(len(self.bufs['acts']) * 0.8)
    if train:
      self.bufs = {key: val[:cut] for key, val in self.bufs.items()}
    else:
      self.bufs = {key: val[cut:] for key, val in self.bufs.items()}

  def __len__(self):
    return len(self.bufs['acts'])

  def __getitem__(self, idx):
    elem = {key: torch.as_tensor(val[idx], dtype=torch.float32) for key, val in self.bufs.items()}
    return elem

def load_ds(C):
  from torchvision import transforms
  train_dset = RolloutDataset(C.datapath, train=True, C=C)
  test_dset = RolloutDataset(C.datapath, train=False, C=C)
  train_loader = DataLoader(train_dset, batch_size=C.bs, shuffle=True, pin_memory=True, num_workers=2)
  test_loader = DataLoader(test_dset, batch_size=C.bs, shuffle=True, pin_memory=True, num_workers=2)
  return train_loader, test_loader
