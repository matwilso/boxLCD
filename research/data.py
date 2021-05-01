from re import I
import uuid
from sys import maxsize
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
from research import utils
from tqdm import tqdm
import time
from research.wrappers import AsyncVectorEnv
BARREL_SIZE = int(1e3)
#from jax.tree_util import tree_map, tree_multimap

def collect(env_fn, G):
  collect_start = time.time()
  G.logdir.mkdir(parents=True, exist_ok=True)
  utils.dump_logger({}, None, 0, G)
  env = env_fn(G)()
  venv = AsyncVectorEnv([env_fn(G) for _ in range(G.num_envs)], G=G)
  assert BARREL_SIZE % G.num_envs == 0, f'barrel size must be divisible by number of envs you use {BARREL_SIZE} % {G.num_envs} != 0'
  assert G.train_barrels != -1 and G.test_barrels != -1, f'must set the number of barrels you want to fill. G.train_barrels=={G.train_barrels}'
  fill_barrels(env, venv, G.test_barrels, 'test', G)
  fill_barrels(env, venv, G.train_barrels, 'train', G)
  print('TOTAL COLLECT TIME', time.time()-collect_start)

def fill_barrels(env, venv, num_barrels, prefix, G):
  """Create files with:
  BARREL_SIZE x EP_LEN x *STATE_DIMS

  o1,a1 --> o2
  Meaning that the last action doesn't matter
  """
  BARS = BARREL_SIZE // G.num_envs
  logdir = G.logdir / prefix
  logdir.mkdir(parents=True, exist_ok=True)
  total_bar = tqdm(total=num_barrels)
  barrel_bar = tqdm(total=BARS)
  total_bar.set_description(f'TOTAL PROGRESS (FPS=N/A)')
  for ti in range(num_barrels):
    obses = {key: np.zeros([BARS, G.num_envs, G.ep_len, *val.shape], dtype=val.dtype) for key, val in env.observation_space.spaces.items()}
    acts = np.zeros([BARS, G.num_envs, G.ep_len, env.action_space.shape[0]])
    barrel_bar.reset()
    for bi in range(BARS):
      start = time.time()
      obs = venv.reset(np.arange(G.num_envs))
      for j in range(G.ep_len):
        act = venv.action_space.sample()
        for key in obses:
          obses[key][bi, :, j] = obs[key]
        acts[bi, :, j] = np.stack(act)
        obs, rew, done, info = venv.step(act)
        # plt.imshow(obs['lcd']);plt.show()
        # venv.render()
        #plt.imshow(1.0*venv.lcd_render()); plt.show()
      barrel_bar.update(1)
      fps = G.ep_len / (time.time() - start)
      barrel_bar.set_description(f'current barrel')
      # barrel_bar.set_description(f'fps: {} | current barrel')
    if (G.logdir / 'pause.marker').exists():
      import ipdb; ipdb.set_trace()

    obses = {key: utils.flatten_first(val) for key, val in obses.items()}
    acts = utils.flatten_first(acts)
    assert obses['proprio'].ndim == 3
    assert acts.ndim == 3
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    data = np.savez_compressed(logdir / f'{timestamp}-{G.ep_len}.barrel', action=acts, **obses)
    total_bar.update(1)
    total_bar.set_description(f'TOTAL PROGRESS (FPS={fps})')


def fill_barrels_slow(env, num_barrels, prefix, G):
  """Create files with:
  BARREL_SIZE x EP_LEN x *STATE_DIMS

  o1,a1 --> o2
  Meaning that the last action doesn't matter
  """
  import ipdb; ipdb.set_trace()
  logdir = G.logdir / prefix
  logdir.mkdir(parents=True, exist_ok=True)
  total_bar = tqdm(total=num_barrels)
  barrel_bar = tqdm(total=BARREL_SIZE)
  total_bar.set_description(f'TOTAL PROGRESS (FPS=N/A)')
  for ti in range(num_barrels):
    obses = {key: np.zeros([BARREL_SIZE, G.ep_len, *val.shape], dtype=val.dtype) for key, val in env.observation_space.spaces.items()}
    acts = np.zeros([BARREL_SIZE, G.ep_len, env.action_space.shape[0]])
    barrel_bar.reset()
    for bi in range(BARREL_SIZE):
      start = time.time()
      obs = env.reset()
      for j in range(G.ep_len):
        act = env.action_space.sample()
        for key in obses:
          obses[key][bi, j] = obs[key]
        acts[bi, j] = act
        obs, rew, done, info = env.step(act)
        # plt.imshow(obs['lcd']);plt.show()
        # env.render()
        #plt.imshow(1.0*env.lcd_render()); plt.show()
      barrel_bar.update(1)
      fps = G.ep_len / (time.time() - start)
      barrel_bar.set_description(f'current barrel')
      # barrel_bar.set_description(f'fps: {} | current barrel')
    if (G.logdir / 'pause.marker').exists():
      import ipdb; ipdb.set_trace()

    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    data = np.savez_compressed(logdir / f'{timestamp}-{G.ep_len}.barrel', action=acts, **obses)
    total_bar.update(1)
    total_bar.set_description(f'TOTAL PROGRESS (FPS={fps})')

class RolloutDataset(IterableDataset):
  def __init__(self, barrel_path, window=int(1e9), infinite=True, refresh_data=False):
    super().__init__()
    self.window = window
    self.infinite = infinite
    self.barrel_path = barrel_path
    self.refresh_data = refresh_data
    self._refresh()

  def _refresh(self):
    """recheck the directory for new barrels"""
    self.barrel_files = list(self.barrel_path.glob('*.barrel.npz'))
    self.nbarrels = len(self.barrel_files)
    assert self.nbarrels > 0, 'didnt find any barrels at datadir'

  def __iter__(self):
    worker_info = th.utils.data.get_worker_info()
    if worker_info is not None:
      np.random.seed(worker_info.id+round(time.time()))

    for ct in itertools.count():
      if self.infinite:
        curr_file = self.barrel_files[np.random.randint(self.nbarrels)]
        if self.refresh_data and ct % 10 == 0:
          self._refresh()
      else:
        curr_file = self.barrel_files[ct]
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
        assert elem['lcd'].max() <= 1.0 and elem['lcd'].min() >= 0.0
        yield elem
      curr_barrel.close()
      if ct >= self.nbarrels - 1 and not self.infinite:
        break

def load_ds(G):
  #def collate_fn(*args, **kwargs):
  #  print(args, kwargs)
  #  tree_multi_map(lambda x, *y: )
  #  import ipdb; ipdb.set_trace()
  #  pass
  train_dset = RolloutDataset(G.datadir / 'train', G.window, refresh_data=G.refresh_data)
  test_dset = RolloutDataset(G.datadir / 'test', G.window, infinite=False)
  train_loader = DataLoader(train_dset, batch_size=G.bs, pin_memory=G.device == 'cuda', num_workers=12, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=G.bs, pin_memory=G.device == 'cuda', num_workers=12, drop_last=True)
  train_loader.nbarrels = train_dset.nbarrels
  test_loader.nbarrels = test_dset.nbarrels
  return train_loader, test_loader
