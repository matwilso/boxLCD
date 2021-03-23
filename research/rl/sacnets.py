from datetime import datetime
import PIL
import argparse
from collections import defaultdict
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Basenet(nn.Module):
  def __init__(self, obs_space, out_size, C):
    super().__init__()
    size = (C.lcd_h*C.lcd_w) // 64
    self.net = nn.Sequential(
        nn.Conv2d(1, C.hidden_size, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(C.hidden_size, C.hidden_size, 3, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(C.hidden_size, C.hidden_size//4, 3, 2, padding=1),
        nn.Flatten(-3)
    )
    self.linear = nn.Linear(size*C.hidden_size//4, out_size)

  def forward(self, obs):
    s, g = obs['lcd'], obs['goal:lcd']
    s = self.net(s[:,None])
    g = self.net(g[:,None])
    x = self.linear(s-g)
    return x

class QFunction(nn.Module):
  def __init__(self, obs_space, act_dim, C):
    super().__init__()
    H = C.hidden_size
    self.base = Basenet(obs_space, H, C)
    self.actpost = nn.Sequential(
        nn.Linear(H + act_dim, H),
        nn.ReLU(),
        nn.Linear(H, 1),
    )
  def forward(self, obs, act):
    x = self.base(obs)
    x = torch.cat([x, act], -1)
    x = self.actpost(x).squeeze(-1)
    return x

class SquashedGaussianActor(nn.Module):
  def __init__(self, obs_space, act_dim, C):
    super().__init__()
    self.net = Basenet(obs_space, 2*act_dim, C)
    self.act_dim = act_dim

  def forward(self, obs, deterministic=False, with_logprob=True):
    net_out = self.net(obs)
    mu, log_std = torch.split(net_out, self.act_dim, dim=-1)

    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)

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

    pi_action = torch.tanh(pi_action)
    return pi_action, logp_pi

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
      self.log_alpha = torch.nn.Parameter(torch.zeros(1))

  def act(self, obs, deterministic=False):
    with torch.no_grad():
      a, _ = self.pi(obs, deterministic, False)
      return a.cpu().numpy()
