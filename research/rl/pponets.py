from re import I
from research import utils
from datetime import datetime
import PIL
import argparse
from collections import defaultdict
from copy import deepcopy
import itertools
import numpy as np
import torch as th
from torch.optim import Adam
import gym
import time
import numpy as np
import scipy.signal
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as thd

def init_weights(m):
  if type(m) == nn.Linear:
    pass
  elif type(m) == nn.Conv2d:
    nn.init.orthogonal_(m.weight)
    m.bias.data.fill_(0.0)

class BaseCNN(nn.Module):
  def __init__(self, obs_space, out_size, G):
    super().__init__()
    size = (G.lcd_h * G.lcd_w) // 64
    self.net = nn.Sequential(
        nn.Conv2d(1, G.nfilter, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(G.nfilter, G.nfilter, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(G.nfilter, G.nfilter, 3, 2, padding=1),
        nn.Flatten(-3)
    )
    mult = 1 if G.zdelta else 2
    #self.linear = nn.Linear(mult * size * G.nfilter, out_size)
    extra = 2 + obs_space['proprio'].shape[0]
    self.linear = nn.Sequential(
        nn.Linear(mult * size * G.nfilter + extra, G.hidden_size),
        nn.ReLU(),
        nn.Linear(G.hidden_size, G.hidden_size),
        nn.ReLU(),
        nn.Linear(G.hidden_size, out_size),
    )
    # nn.init.zeros_(self.linear[-1].weight)
    nn.init.zeros_(self.linear[-1].bias)
    # self.net.apply(init_weights)
    self.G = G

  def forward(self, obs):
    assert obs['lcd'].max().detach().cpu() <= 1.0
    s, g = obs['lcd'], obs['goal:lcd']
    s = self.net(s[:, None])
    g = self.net(g[:, None])
    if self.G.zdelta:
      #x = s
      x = g - s
    else:
      x = th.cat([s, g], -1)
    x = th.cat([x, obs['goal:proprio'], obs['proprio']], -1)
    x = self.linear(x)
    return x

class BaseCMLP(nn.Module):
  def __init__(self, obs_space, out_size, G):
    super().__init__()
    size = G.lcd_h * G.lcd_w
    self.net = nn.Sequential(
        nn.Linear(size, G.hidden_size),
        nn.ReLU(),
        nn.Linear(G.hidden_size, G.hidden_size),
        nn.ReLU(),
        nn.Linear(G.hidden_size, G.hidden_size),
    )
    mult = 1 if G.zdelta else 2
    self.linear = nn.Linear(mult * G.hidden_size, out_size)
    self.G = G

  def forward(self, obs):
    s, g = obs['lcd'], obs['goal:lcd']
    s = self.net(s.flatten(-2))
    g = self.net(g.flatten(-2))
    if self.G.zdelta:
      x = g - s
      #x = s
    else:
      x = th.cat([s, g], -1)
    x = self.linear(x)
    return x

class BaseMLP(nn.Module):
  def __init__(self, in_size, out_size, G):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_size, G.hidden_size),
        nn.ReLU(),
        nn.Linear(G.hidden_size, G.hidden_size),
        nn.ReLU(),
        nn.Linear(G.hidden_size, G.hidden_size),
        nn.ReLU(),
        nn.Linear(G.hidden_size, out_size),
    )

  def forward(self, x):
    return self.net(x)

class Actor(nn.Module):
  def __init__(self, obs_space, act_dim, G):
    super().__init__()
    log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
    self.log_std = th.nn.Parameter(th.as_tensor(log_std))
    gsize = obs_space['goal:proprio'].shape[0]
    size = obs_space[G.state_key].shape[0] + gsize
    self.size = size
    self.net = BaseMLP(size, act_dim, G)
    self.G = G

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act).sum(axis=-1)

  def forward(self, obs, act=None):
    pi = self._distribution(obs)
    logp_a = None
    if act is not None:
      logp_a = self._log_prob_from_distribution(pi, act)
    return pi, logp_a

  def _distribution(self, obs):
    x = th.cat([obs[self.G.state_key], obs['goal:proprio']], -1)
    mu = self.net(x)
    std = th.exp(self.log_std)
    return thd.Normal(mu, std)

class Critic(nn.Module):
  def __init__(self, obs_space, act_dim, G, preproc=None):
    super().__init__()
    self.G = G
    gsize = obs_space['goal:proprio'].shape[0]
    size = obs_space[self.G.state_key].shape[0] + gsize
    self.base = BaseMLP(size, 1, G)

  def forward(self, obs):
    x = th.cat([obs[self.G.state_key], obs['goal:proprio']], -1)
    return self.base(x).squeeze(-1)

class ActorCritic(nn.Module):
  def __init__(self, obs_space, act_space, G=None):
    super().__init__()
    act_dim = act_space.shape[0]
    self.G = G
    self.pi = Actor(obs_space, act_dim, G)
    # build value function
    self.v = Critic(obs_space, act_dim, G)

  def step(self, obs):
    obs = {key: th.as_tensor(1.0 * val, dtype=th.float32).to(self.G.device) for key, val in obs.items()}
    with th.no_grad():
      pi = self.pi._distribution(obs)
      a = pi.sample()
      logp_a = self.pi._log_prob_from_distribution(pi, a)
      v = self.v(obs)
    if self.G.lenv:
      return a, v, logp_a
    else:
      return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

  def act(self, obs):
    return self.step(obs)[0]