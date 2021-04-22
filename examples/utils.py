import torch as th
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
import argparse
from collections import defaultdict
import subprocess
import sys
import pathlib
from boxLCD import envs, env_map, ENV_DG
from boxLCD.utils import args_type, AttrDict

def config():
  # G as in confi(G), fla(G), settin(G), ar(G). G is a single letter that is not overloaded already ((F)unctional, (C)hannel, (H)eight, etc.)
  G = AttrDict()
  # BASICS
  G.logdir = pathlib.Path('./logs/')
  G.datapath = pathlib.Path('.')
  G.collect_n = 10000
  G.env = 'Bounce'
  G.lcd_mode = '1'  # just for visualization
  # training stuff
  G.device = 'cuda'  # 'cuda', 'cpu'
  G.num_epochs = 200
  G.bs = 64
  G.lr = 5e-4
  G.n_layer = 2
  G.n_embed = 128
  G.n_head = 4
  # extra info that we set here for convenience and don't modify
  G.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
  G.commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
  pastKeys = list(G.keys())
  for key, val in ENV_DG.items():
    assert key not in pastKeys, f'make sure you are not duplicating keys {key}'
    G[key] = val
  return G

def parse_args():
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  tempC = parser.parse_args()
  Env = env_map[tempC.env]
  parser.set_defaults(**Env.ENV_DG)
  G = AttrDict(parser.parse_args().__dict__)
  return G

class RolloutDataset(Dataset):
  def __init__(self, npzfile, train=True, G=None):
    data = np.load(npzfile, allow_pickle=True)
    self.bufs = {key: th.as_tensor(data[key]) for key in data.keys()}
    cut = int(len(self.bufs['action']) * 0.8)
    if train:
      self.bufs = {key: val[:cut] for key, val in self.bufs.items()}
    else:
      self.bufs = {key: val[cut:] for key, val in self.bufs.items()}

  def __len__(self):
    return len(self.bufs['action'])

  def __getitem__(self, idx):
    elem = {key: th.as_tensor(val[idx], dtype=th.float32) for key, val in self.bufs.items()}
    elem['lcd'] /= 255.0
    return elem

def load_ds(G):
  train_dset = RolloutDataset(G.datapath, train=True, G=G)
  test_dset = RolloutDataset(G.datapath, train=False, G=G)
  train_loader = DataLoader(train_dset, batch_size=G.bs, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=G.bs, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
  return train_loader, test_loader

def dump_logger(logger, writer, i, G):
  print('=' * 30)
  print(i)
  for key in logger:
    val = np.mean(logger[key])
    if writer is not None:
      writer.add_scalar(key, val, i)
    print(key, val)
  print(G.full_cmd)
  print(G.num_vars)
  pathlib.Path(G.logdir).mkdir(parents=True, exist_ok=True)
  with open(pathlib.Path(G.logdir) / 'hps.yaml', 'w') as f:
    yaml.dump(dict(G), f, width=1000)
  print('=' * 30)
  return defaultdict(lambda: [])

def count_vars(module):
  return sum([np.prod(p.shape) for p in module.parameters()])

def force_shape(out):
  """take one right before video and force it's shape"""
  N, T, C, H, W = out.shape
  if isinstance(out, np.ndarray):
    out = out.transpose(1, 2, 3, 0, 4)
    out = np.concatenate([out, np.zeros(out.shape[:-1], dtype=out.dtype)[..., None]], -1)
  else:
    out = out.permute(1, 2, 3, 0, 4)
    out = th.cat([out, th.zeros(out.shape[:-1])[..., None]], -1)
  out = out.reshape(T, C, H, N * (W + 1))[None]
  return out
