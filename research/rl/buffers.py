from collections import defaultdict
import torch as th
import numpy as np
from research import utils
from jax.tree_util import tree_multimap, tree_map

def combined_shape(size, shape):
  return (size, *shape)

class OGRB:
  """
  A simple FIFO experience replay buffer for SAC agents.
  """
  def __init__(self, C, obs_space, act_space):
    self.C = C
    size = C.replay_size
    self.bufs = {}
    for x in ['o:', 'o2:']:
      for key in obs_space.spaces:
        self.bufs[x + key] = np.zeros((size, *obs_space.spaces[key].shape), dtype=np.float32)
    self.bufs['act'] = np.zeros((size, *act_space.shape), dtype=np.float32)
    self.bufs['rew'] = np.zeros(size, dtype=np.float32)
    self.bufs['done'] = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size
    self.ptrs = [0]

  #def mark_done(self):
  #  self.ptrs += [self.ptr]

  def store(self, ntrans):
    for key in self.bufs:
      self.bufs[key][self.ptr] = ntrans[key]
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  #def get_last(self, n):
  #  assert self.C.ep_len*n <= self.ptr
  #  batch = tree_map(lambda x: x[self.ptrs[-2]:self.ptrs[-1]], self.bufs)
  #  batch = tree_multimap(lambda x, y: np.concatenate([x, y[self.ptrs[-3]:self.ptrs[-2]]]), batch, self.bufs)
  #  batch = tree_multimap(lambda x, y: np.concatenate([x, y[self.ptrs[-4]:self.ptrs[-3]]]), batch, self.bufs)
  #  batch = tree_multimap(lambda x, y: np.concatenate([x, y[self.ptrs[-5]:self.ptrs[-4]]]), batch, self.bufs)
  #  o = utils.filtdict(batch, 'o:', fkey=lambda x: x[2:])
  #  o2 = utils.filtdict(batch, 'o2:', fkey=lambda x: x[3:])
  #  batch = utils.nfiltdict(batch, '(o:|o2:)')
  #  batch['obs'] = o
  #  batch['obs2'] = o2
  #  return batch


  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    batch = tree_map(lambda x: x[idxs], self.bufs)
    o = utils.filtdict(batch, 'o:', fkey=lambda x: x[2:])
    o2 = utils.filtdict(batch, 'o2:', fkey=lambda x: x[3:])
    batch = utils.nfiltdict(batch, '(o:|o2:)')
    batch['obs'] = o
    batch['obs2'] = o2
    assert np.isclose(np.mean((o['goal:pstate'] - o2['goal:pstate'])**2), 0.0), "AHH"
    return tree_map(lambda v: th.as_tensor(v, dtype=th.float32).to(self.C.device), batch)

class ReplayBuffer:
  """
  A simple FIFO experience replay buffer for SAC agents.
  """
  def __init__(self, C, obs_space, act_space):
    self.C = C
    size = C.replay_size
    self.bufs = {}
    for o in ['o', 'o2']:
      for key in obs_space.spaces:
        self.bufs[o + ':' + key] = np.zeros((size, *obs_space.spaces[key].shape), dtype=np.float32)
    self.bufs['act'] = np.zeros((size, *act_space.shape), dtype=np.float32)
    self.bufs['rew'] = np.zeros(size, dtype=np.float32)
    self.bufs['done'] = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store_n(self, ntrans):
    shape = self.C.num_envs
    end_idx = self.ptr + shape
    if end_idx <= self.max_size:  # normal operation
      for key in self.bufs:
        self.bufs[key][self.ptr:end_idx] = ntrans[key]
      self.ptr = (self.ptr + shape) % self.max_size
    else:  # handle wrap around
      overflow = (end_idx - self.max_size)
      top_off = shape - overflow
      for key in self.bufs:
        self.bufs[key][self.ptr:self.ptr + top_off] = ntrans[key][:top_off]  # top off the last end of the array
        self.bufs[key][:overflow] = ntrans[key][top_off:]  # start over at beginning
      self.ptr = overflow
    self.size = min(self.size + shape, self.max_size)

  def sample_batch(self, batch_size=32):
    #import matplotlib.pyplot as plt
    #for i in range(800)[::8]:
    #  plt.imsave(f'test{i}.png', self.bufs['o:lcd'][i][...,None].repeat(3,-1))
    idxs = np.random.randint(0, self.size, size=batch_size)
    batch = tree_map(lambda x: x[idxs], self.bufs)
    o = utils.filtdict(batch, 'o:', fkey=lambda x: x[2:])
    o2 = utils.filtdict(batch, 'o2:', fkey=lambda x: x[3:])
    batch = utils.nfiltdict(batch, '(o:|o2:)')
    batch['obs'] = o
    batch['obs2'] = o2
    assert np.isclose(np.mean((o['goal:pstate'] - o2['goal:pstate'])**2), 0.0), "AHH"
    return tree_map(lambda v: th.as_tensor(v, dtype=th.float32).to(self.C.device), batch)
