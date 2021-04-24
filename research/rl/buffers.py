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
  def __init__(self, G, obs_space, act_space):
    self.G = G
    size = G.replay_size
    self.bufs = {}
    for x in ['o:', 'o2:']:
      for key in obs_space.spaces:
        self.bufs[x + key] = np.zeros((size, *obs_space.spaces[key].shape), dtype=np.float32)
    self.bufs['act'] = np.zeros((size, *act_space.shape), dtype=np.float32)
    self.bufs['rew'] = np.zeros(size, dtype=np.float32)
    self.bufs['done'] = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size
    self.ptrs = [0]

  # def mark_done(self):
  #  self.ptrs += [self.ptr]

  def store(self, ntrans):
    for key in self.bufs:
      self.bufs[key][self.ptr] = ntrans[key]
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  # def get_last(self, n):
  #  assert self.G.ep_len*n <= self.ptr
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
    assert np.isclose(np.mean((o['goal:proprio'] - o2['goal:proprio'])**2), 0.0), "AHH"
    return tree_map(lambda v: th.as_tensor(v, dtype=th.float32).to(self.G.device), batch)

class ReplayBuffer:
  """
  A simple FIFO experience replay buffer for SAC agents.
  """
  def __init__(self, G, obs_space, act_space):
    self.G = G
    size = G.replay_size
    self.bufs = {}
    for o in ['o', 'o2']:
      for key in obs_space.spaces:
        self.bufs[o + ':' + key] = np.zeros((size, *obs_space.spaces[key].shape), dtype=np.float32)
    self.bufs['act'] = np.zeros((size, *act_space.shape), dtype=np.float32)
    self.bufs['rew'] = np.zeros(size, dtype=np.float32)
    self.bufs['done'] = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size
    if G.lenv:
      self.proc = lambda x: x.cpu().numpy()
    else:
      self.proc = lambda x: x

  def store_n(self, ntrans):
    shape = self.G.num_envs
    end_idx = self.ptr + shape
    if end_idx <= self.max_size:  # normal operation
      for key in self.bufs:
        self.bufs[key][self.ptr:end_idx] = self.proc(ntrans[key])
      self.ptr = (self.ptr + shape) % self.max_size
    else:  # handle wrap around
      overflow = (end_idx - self.max_size)
      top_off = shape - overflow
      for key in self.bufs:
        self.bufs[key][self.ptr:self.ptr + top_off] = self.proc(ntrans[key][:top_off])  # top off the last end of the array
        self.bufs[key][:overflow] = self.proc(ntrans[key][top_off:])  # start over at beginning
      self.ptr = overflow
    self.size = min(self.size + shape, self.max_size)

  def sample_batch(self, batch_size=32):
    #import matplotlib.pyplot as plt
    # for i in range(800)[::8]:
    #  plt.imsave(f'test{i}.png', self.bufs['o:lcd'][i][...,None].repeat(3,-1))
    idxs = np.random.randint(0, self.size, size=batch_size)
    batch = tree_map(lambda x: x[idxs], self.bufs)
    o = utils.filtdict(batch, 'o:', fkey=lambda x: x[2:])
    o2 = utils.filtdict(batch, 'o2:', fkey=lambda x: x[3:])
    batch = utils.nfiltdict(batch, '(o:|o2:)')
    batch['obs'] = o
    batch['obs2'] = o2
    assert np.isclose(np.mean((o['goal:proprio'] - o2['goal:proprio'])**2), 0.0), "AHH"
    return tree_map(lambda v: th.as_tensor(v, dtype=th.float32).to(self.G.device), batch)


class PPOBuffer:
  """
  A buffer for storing trajectories experienced by a PPO agent interacting
  with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
  for calculating the advantages of state-action pairs.
  """

  def __init__(self, config, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
    self.bufs = {}
    self.bufs['obs'] = np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
    self.bufs['act'] = np.zeros(utils.combined_shape(size, act_dim), dtype=np.float32)
    self.bufs['adv'] = np.zeros(size, dtype=np.float32)
    self.bufs['rew'] = np.zeros(size, dtype=np.float32)
    self.bufs['ret'] = np.zeros(size, dtype=np.float32)
    self.bufs['val'] = np.zeros(size, dtype=np.float32)
    self.bufs['logp'] = np.zeros(size, dtype=np.float32)
    self.gamma, self.lam = gamma, lam
    self.ptr, self.path_start_idx, self.max_size = 0, np.zeros(config.num_envs), size
    self.trajs = [defaultdict(lambda: []) for _ in range(config.num_envs)]
    self.config = config

  def store_n(self, ntrans):
    #shape = ntrans['obs'].shape[0]
    # assert self.ptr+shape< self.max_size     # buffer has to have room so you can store
    for key in ntrans:
      for idx in range(self.config.num_envs):
        self.trajs[idx][key] += [ntrans[key][idx]]

  def finish_paths(self, idxs, last_vals):
    """
    Call this at the end of a trajectory, or when one gets cut off
    by an epoch ending. This looks back in the buffer to where the
    trajectory started, and uses rewards and value estimates from
    the whole trajectory to compute advantage estimates with GAE-Lambda,
    as well as compute the rewards-to-go for each state, to use as
    the targets for the value function.
    The "last_val" argument should be 0 if the trajectory ended
    because the agent reached a terminal state (died), and otherwise
    should be V(s_T), the value function estimated for the last state.
    This allows us to bootstrap the reward-to-go calculation to account
    for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
    """
    # assert self.ptr+shape< self.max_size     # buffer has to have room so you can store
    for idx in idxs:
      size = len(self.trajs[idx]['obs'])
      rews = np.array(self.trajs[idx]['rew'] + [last_vals[idx]])
      vals = np.array(self.trajs[idx]['val'] + [last_vals[idx]])

      # the next two lines implement GAE-Lambda advantage calculation
      deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
      self.trajs[idx]['adv'] = utils.discount_cumsum(deltas, self.gamma * self.lam)

      # the next line computes rewards-to-go, to be targets for the value function
      self.trajs[idx]['ret'] = utils.discount_cumsum(rews, self.gamma)[:-1]

      for key in self.trajs[idx]:
        self.bufs[key][self.ptr:self.ptr + size] = self.trajs[idx][key]
      self.ptr += size
      self.trajs[idx] = defaultdict(lambda: [])

  def get(self):
    """
    Call this at the end of an epoch to get all of the data from
    the buffer, with advantages appropriately normalized (shifted to have
    mean zero and std one). Also, resets some pointers in the buffer.
    """
    assert self.ptr == self.max_size    # buffer has to be full before you can get
    self.ptr, self.path_start_idx = 0, 0
    # the next two lines implement the advantage normalization trick
    adv_mean, adv_std = np.mean(self.bufs['adv']), np.std(self.bufs['adv'])
    self.bufs['adv'] = (self.bufs['adv'] - adv_mean) / adv_std
    data = utils.subdict(self.bufs, ['obs', 'act', 'ret', 'adv', 'logp'])
    return {k: th.as_tensor(v, dtype=th.float32).to(self.config.device) for k, v in data.items()}
