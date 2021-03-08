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
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from boxLCD.utils import A
import utils

class RolloutDataset(Dataset):
  def __init__(self, npzfile, train=True, C=None):
    data = np.load(npzfile, allow_pickle=True)
    self.bufs = {key: torch.as_tensor(data[key]) for key in data.keys()}
    if C.datamode == 'image':
      self.bufs = {key: val.flatten(0, 1) for key, val in self.bufs.items()}
      self.bufs['lcd'] = self.bufs['lcd'][:,None]
    cut = int(len(self.bufs['acts']) * 0.8)
    if train:
      self.bufs = {key: val[:cut] for key, val in self.bufs.items()}
    else:
      self.bufs = {key: val[cut:] for key, val in self.bufs.items()}
    self.C = C

  def __len__(self):
    return len(self.bufs['acts'])

  def __getitem__(self, idx):
    start = np.random.randint(0, self.C.ep_len-self.C.window)
    if self.C.datamode == 'video' and self.C.vidstack != self.C.ep_len:
      elem = {key: torch.as_tensor(val[idx, start:start+self.C.window], dtype=torch.float32) for key, val in self.bufs.items()}
    else:
      elem = {key: torch.as_tensor(val[idx], dtype=torch.float32) for key, val in self.bufs.items()}
    elem['lcd'] /= 255.0
    return elem

def load_ds(C):
  train_dset = RolloutDataset(C.datapath, train=True, C=C)
  test_dset = RolloutDataset(C.datapath, train=False, C=C)
  #train_dset = RolloutDataset(C.datapath / 'dump.npz', train=True, C=C)
  #test_dset = RolloutDataset(C.datapath / 'dump.npz', train=False, C=C)
  train_loader = DataLoader(train_dset, batch_size=C.bs, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=C.bs, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
  return train_loader, test_loader