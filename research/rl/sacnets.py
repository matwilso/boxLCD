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
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def init_weights(m):
  if type(m) == nn.Linear:
    pass
  elif type(m) == nn.Conv2d:
    nn.init.orthogonal_(m.weight)
    m.bias.data.fill_(0.0)

class BaseCNN(nn.Module):
  def __init__(self, obs_space, out_size, C):
    super().__init__()
    size = (C.lcd_h * C.lcd_w) // 64
    self.net = nn.Sequential(
        nn.Conv2d(1, C.nfilter, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(C.nfilter, C.nfilter, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(C.nfilter, C.nfilter, 3, 2, padding=1),
        nn.Flatten(-3)
    )
    mult = 1 if C.zdelta else 2
    #self.linear = nn.Linear(mult * size * C.nfilter, out_size)
    self.linear = nn.Sequential(
      nn.Linear(mult * size * C.nfilter, C.hidden_size),
      nn.ReLU(),
      nn.Linear(C.hidden_size, out_size),
    )
    #nn.init.zeros_(self.linear[-1].weight)
    nn.init.zeros_(self.linear[-1].bias)
    #self.net.apply(init_weights)
    self.C = C

  def forward(self, obs):
    assert obs['lcd'].max().detach().cpu() <= 1.0
    s, g = obs['lcd'], obs['goal:lcd']
    #s = self.net(s[:, None])
    #g = self.net(g[:, None])
    s = self.net(obs['lcd'][:,None])
    if self.C.zdelta:
      x = s
      #x = g - s
    else:
      x = th.cat([s, g], -1)
    x = self.linear(x)
    return x

class BaseCMLP(nn.Module):
  def __init__(self, obs_space, out_size, C):
    super().__init__()
    size = C.lcd_h * C.lcd_w
    self.net = nn.Sequential(
        nn.Linear(size, C.hidden_size),
        nn.ReLU(),
        nn.Linear(C.hidden_size, C.hidden_size),
        nn.ReLU(),
        nn.Linear(C.hidden_size, C.hidden_size),
    )
    mult = 1 if C.zdelta else 2
    self.linear = nn.Linear(mult * C.hidden_size, out_size)
    self.C = C

  def forward(self, obs):
    s, g = obs['lcd'], obs['goal:lcd']
    s = self.net(s.flatten(-2))
    g = self.net(g.flatten(-2))
    if self.C.zdelta:
      x = s
      #x = g- s
    else:
      x = th.cat([s, g], -1)
    x = self.linear(x)
    return x


class BaseMLP(nn.Module):
  def __init__(self, in_size, out_size, C):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_size, C.hidden_size),
        nn.ReLU(),
        nn.Linear(C.hidden_size, C.hidden_size),
        nn.ReLU(),
        nn.Linear(C.hidden_size, C.hidden_size),
        nn.ReLU(),
        nn.Linear(C.hidden_size, out_size),
    )

  def forward(self, x):
    return self.net(x)

class QFunction(nn.Module):
  def __init__(self, obs_space, act_dim, C):
    super().__init__()
    H = C.hidden_size
    self.C = C
    size = obs_space['pstate'].shape[0] * 2 + act_dim
    if self.C.net == 'mlp':
      self.base = BaseMLP(size, 1, C)
    elif self.C.net == 'cmlp':
      self.base = BaseCMLP(obs_space, H, C)
    elif self.C.net == 'cnn':
      self.base = BaseCNN(obs_space, H, C)
    self.actin = nn.Linear(act_dim, H)
    self.act_head = nn.Sequential(
        nn.Linear(2 * H, H),
        nn.ReLU(),
        nn.Linear(H, 1),
    )

  def forward(self, obs, act):
    if self.C.net == 'mlp':
      x = th.cat([obs['pstate'], obs['goal:pstate'], act], -1)
      return self.base(x).squeeze(-1)
    else:
      x = self.base(obs)
      act = self.actin(act)
      x = th.cat([x, act], -1)
      x = self.act_head(x)
      return x.squeeze(-1)

class SquashedGaussianActor(nn.Module):
  def __init__(self, obs_space, act_dim, C):
    super().__init__()
    self.C = C
    size = obs_space['pstate'].shape[0] * 2
    if self.C.net == 'mlp':
      self.net = BaseMLP(size, 2 * act_dim, C)
    elif self.C.net == 'cmlp':
      self.net = BaseCMLP(obs_space, 2 * act_dim, C)
    elif self.C.net == 'cnn':
      self.net = BaseCNN(obs_space, 2 * act_dim, C)
    self.act_dim = act_dim

  def forward(self, obs, deterministic=False, with_logprob=True):
    if self.C.net == 'mlp':
      obs = th.cat([obs['pstate'], obs['goal:pstate']], -1)
    net_out = self.net(obs)
    mu, log_std = th.split(net_out, self.act_dim, dim=-1)

    log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = th.exp(log_std)

    # Pre-squash distribution and sample
    pi_distribution = Normal(mu, std)
    if deterministic:
      # Only used for evaluating policy at test time.
      pi_action = mu
    else:
      pi_action = pi_distribution.rsample()

    if with_logprob:
      # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
      # NOTE: The correction formula is a little bit magic. To get an understanding
      # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
      # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
      # Try deriving it yourself as a (very difficult) exercise. :)
      logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
      logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
    else:
      logp_pi = None

    pi_action = th.tanh(pi_action)
    return pi_action, logp_pi, {'mean': th.tanh(mu), 'std': std}

class ActorCritic(nn.Module):
  def __init__(self, obs_space, act_space, C=None):
    super().__init__()
    act_dim = act_space.shape[0]
    # build policy and value functions
    self.pi = SquashedGaussianActor(obs_space, act_dim, C=C)
    self.q1 = QFunction(obs_space, act_dim, C=C)
    self.q2 = QFunction(obs_space, act_dim, C=C)
    if C.learned_alpha:
      self.target_entropy = -np.prod(act_space.shape)
      self.log_alpha = th.nn.Parameter(th.zeros(1))

  def act(self, obs, deterministic=False):
    with th.no_grad():
      a, _, ainfo = self.pi(obs, deterministic, False)
      return a.cpu().numpy()
