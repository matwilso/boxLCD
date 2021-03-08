import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from boxLCD import envs
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
import pathlib
from collections import defaultdict

class RolloutDataset(Dataset):
  def __init__(self, npzfile, train=True, C=None):
    data = np.load(npzfile, allow_pickle=True)
    self.bufs = {key: torch.as_tensor(data[key]) for key in data.keys()}
    cut = int(len(self.bufs['acts']) * 0.8)
    if train:
      self.bufs = {key: val[:cut] for key, val in self.bufs.items()}
    else:
      self.bufs = {key: val[cut:] for key, val in self.bufs.items()}

  def __len__(self):
    return len(self.bufs['acts'])

  def __getitem__(self, idx):
    elem = {key: torch.as_tensor(val[idx], dtype=torch.float32) for key, val in self.bufs.items()}
    elem['lcd'] /= 255.0
    return elem

def load_ds(C):
  train_dset = RolloutDataset(C.datapath, train=True, C=C)
  test_dset = RolloutDataset(C.datapath, train=False, C=C)
  train_loader = DataLoader(train_dset, batch_size=C.bs, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=C.bs, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
  return train_loader, test_loader

def dump_logger(logger, writer, i, C):
  print('=' * 30)
  print(i)
  for key in logger:
    val = np.mean(logger[key])
    if writer is not None:
      writer.add_scalar(key, val, i)
    print(key, val)
  print(C.full_cmd)
  print(C.num_vars)
  pathlib.Path(C.logdir).mkdir(parents=True, exist_ok=True)
  with open(pathlib.Path(C.logdir) / 'hps.yaml', 'w') as f:
    yaml.dump(dict(C), f)
  print('=' * 30)
  return defaultdict(lambda: [])

def count_vars(module):
  return sum([np.prod(p.shape) for p in module.parameters()])

def force_shape(out):
  """take one right before video and force it's shape"""
  N, T, C, H, W = out.shape
  if isinstance(out, np.ndarray):
    out = out.transpose(1, 2, 3, 0, 4)
    out = np.concatenate([out,np.zeros(out.shape[:-1], dtype=out.dtype)[...,None]], -1)
  else:
    out = out.permute(1, 2, 3, 0, 4)
    out = torch.cat([out,torch.zeros(out.shape[:-1])[...,None]], -1)
  out = out.reshape(T, C, H, N * (W+1))[None]
  return out
