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

from nets.core import mlp, MLPQFunction

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianActor(nn.Module):
    def __init__(self, tenv, net, obs_dim, act_dim, hidden_sizes, activation, act_limit, cfg=None):
        super().__init__()
        if net == 'mlp':
            self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim*2], activation)
        self.act_limit = act_limit
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
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class ActorCritic(nn.Module):

    def __init__(self, tenv, observation_space, action_space, hidden_sizes=(256,256), activation=nn.ReLU, cfg=None):
        super().__init__()
        import ipdb; ipdb.set_trace()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianActor(tenv, cfg.net, obs_dim, act_dim, hidden_sizes, activation, act_limit, cfg=cfg)
        if cfg.net == 'mlp':
            self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
            self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

        if cfg.learned_alpha:
            self.target_entropy = -np.prod(action_space.shape)
            self.log_alpha = torch.nn.Parameter(torch.zeros(1))

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()