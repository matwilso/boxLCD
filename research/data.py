import uuid
from sys import maxsize
from sync_vector_env import SyncVectorEnv
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
import torch as th
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import Adam
import numpy as np
import yaml
from datetime import datetime
import argparse
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from boxLCD.utils import A
import utils
from tqdm import tqdm
import time

BARREL_SIZE = int(1e3)

def collect(make_env, C):
  C.logdir.mkdir(parents=True, exist_ok=True)
  utils.dump_logger({}, None, 0, C)
  env = make_env()
  NB = C.num_barrels
  total_bar = tqdm(total=NB)
  barrel_bar = tqdm(total=BARREL_SIZE)
  total_bar.set_description(f'TOTAL PROGRESS (FPS=N/A)')
  for ti in range(NB):
    obses = {key: np.zeros([BARREL_SIZE, C.ep_len, *val.shape], dtype=val.dtype) for key, val in env.observation_space.spaces.items()}
    acts = np.zeros([BARREL_SIZE, C.ep_len, env.action_space.shape[0]])
    barrel_bar.reset()
    for bi in range(BARREL_SIZE):
      start = time.time()
      obs = env.reset()
      for j in range(C.ep_len):
        act = env.action_space.sample()
        for key in obses:
          obses[key][bi, j] = obs[key]
        acts[bi, j] = act
        obs, rew, done, info = env.step(act)
        # plt.imshow(obs['lcd']);plt.show()
        # env.render()
        #plt.imshow(1.0*env.lcd_render()); plt.show()
      barrel_bar.update(1)
      fps = C.ep_len / (time.time() - start)
      barrel_bar.set_description(f'current barrel')
      # barrel_bar.set_description(f'fps: {} | current barrel')
    if (C.logdir / 'pause.marker').exists():
      import ipdb; ipdb.set_trace()

    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    data = np.savez_compressed(f'{C.logdir}/{timestamp}-{C.ep_len}.barrel', acts=acts, **obses)
    total_bar.update(1)
    total_bar.set_description(f'TOTAL PROGRESS (FPS={fps})')

class RolloutDataset(IterableDataset):
  def __init__(self, barrel_path, window=int(1e9), infinite=True):
    super().__init__()
    self.barrel_files = list(barrel_path.glob('*.barrel.npz'))
    self.nbarrels = len(self.barrel_files)
    self.window = window
    self.infinite = infinite

  def __iter__(self):
    worker_info = th.utils.data.get_worker_info()
    if worker_info is not None:
      np.random.seed(worker_info.id)

    ct = 0
    while True:
      if self.infinite:
        curr_file = self.barrel_files[np.random.randint(self.nbarrels)]
      else:
        curr_file = self.barrel_files[ct]
        ct += 1
      curr_barrel = np.load(curr_file, allow_pickle=True)
      elems = {key: th.as_tensor(curr_barrel[key], dtype=th.float32) for key in curr_barrel.keys()}
      idxs = np.arange(BARREL_SIZE)
      np.random.shuffle(idxs)
      max_start = elems['lcd'].shape[1] - self.window
      for idx in idxs:
        if max_start > 0:
          start = np.random.randint(0, max_start)
          elem = {key: th.as_tensor(val[idx, start:start + self.window], dtype=th.float32) for key, val in elems.items()}
        else:
          elem = {key: th.as_tensor(val[idx], dtype=th.float32) for key, val in elems.items()}
        elem['lcd'] /= 255.0
        yield elem
      curr_barrel.close()
      if ct >= self.nbarrels and not self.infinite:
        break

def load_ds(C):
  train_dset = RolloutDataset(C.datapath / 'train', C.window)
  test_dset = RolloutDataset(C.datapath / 'test', C.window, infinite=False)
  train_loader = DataLoader(train_dset, batch_size=C.bs, pin_memory=C.device == 'cuda', num_workers=8, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=C.bs, pin_memory=C.device == 'cuda', num_workers=8, drop_last=True)
  return train_loader, test_loader