import itertools

import numpy as np
import PIL
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from research import utils
from research.wrappers.cube_goal import CubeGoalEnv

LOG_STD_MAX = 2
LOG_STD_MIN = -20


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
            nn.Flatten(-3),
        )
        mult = 1 if G.zdelta else 2
        # self.linear = nn.Linear(mult * size * G.nfilter, out_size)
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
            # x = s
            x = g - s
        else:
            x = torch.cat([s, g], -1)
        x = torch.cat([x, obs['goal:compact'], obs['proprio']], -1)
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
            # x = s
        else:
            x = torch.cat([s, g], -1)
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


class QFunction(nn.Module):
    def __init__(self, obs_space, act_dim, goal_key, G, preproc=None):
        super().__init__()
        H = G.hidden_size
        self.G = G
        gsize = obs_space[goal_key].shape[0]
        self.goal_key = goal_key
        size = obs_space[self.G.state_key].shape[0] + gsize + act_dim
        if self.G.net == 'mlp':
            self.base = BaseMLP(size, 1, G)
        elif self.G.net == 'cmlp':
            self.base = BaseCMLP(obs_space, H, G)
        elif self.G.net == 'cnn':
            self.base = BaseCNN(obs_space, H, G)
        self.actin = nn.Linear(act_dim, H)
        self.act_head = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

    def forward(self, obs, act):
        if self.G.net == 'mlp':
            x = torch.cat([obs[self.G.state_key], obs[self.goal_key], act], -1)
            return self.base(x).squeeze(-1)
        elif self.G.net == 'bvae':
            x = self.preproc.encode(obs).detach()
            x = self.statie(x)
            if True:
                # goals = utils.filtdict(obs, 'goal:', fkey=lambda x: x[5:])
                # gx = self.preproc.encode(goals).detach()
                # gx = self.goalie(gx)
                x = torch.cat([x, obs[self.goal_key]], -1)
                # x = x + gx
            xa = self.actin(act)
            x = torch.cat([x, xa], -1)
            x = self.act_head(x)
            return x.squeeze(-1)
        else:
            x = self.base(obs)
            act = self.actin(act)
            x = torch.cat([x, act], -1)
            x = self.act_head(x)
            return x.squeeze(-1)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_space, act_dim, goal_key, G, preproc=None):
        super().__init__()
        self.G = G
        gsize = obs_space[goal_key].shape[0]
        self.goal_key = goal_key
        size = obs_space[self.G.state_key].shape[0] + gsize
        self.size = size
        if self.G.net == 'mlp':
            self.net = BaseMLP(size, 2 * act_dim, G)
        elif self.G.net == 'cmlp':
            self.net = BaseCMLP(obs_space, 2 * act_dim, G)
        elif self.G.net == 'cnn':
            self.net = BaseCNN(obs_space, 2 * act_dim, G)
        elif 'bvae' in self.G.net:
            self.preproc = preproc
            # self.goalie = nn.Linear(self.preproc.z_size, G.hidden_size//2)
            self.statie = nn.Linear(self.preproc.z_size, G.hidden_size)
            self.net = nn.Sequential(
                nn.Linear(G.hidden_size + gsize, G.hidden_size),
                nn.ReLU(),
                # nn.Linear(G.hidden_size, G.hidden_size),
                # nn.ReLU(),
                nn.Linear(G.hidden_size, G.hidden_size),
                nn.ReLU(),
                nn.Linear(G.hidden_size, 2 * act_dim),
            )
        self.act_dim = act_dim

    def forward(self, obs, deterministic=False, with_logprob=True):
        if self.G.net == 'mlp':
            x = torch.cat([obs[self.G.state_key], obs[self.goal_key]], -1)
        elif self.G.net == 'bvae':
            z = self.preproc.encode(obs).detach()
            x = self.statie(z)
            if True:
                # goals = utils.filtdict(obs, 'goal:', fkey=lambda x: x[5:])
                # gz = self.preproc.encode(goals).detach()
                # gx = self.goalie(gz)
                # x = torch.cat([x, gx], -1)
                x = torch.cat([x, obs[self.goal_key]], -1)
                # 1 - torch.logical_and(z, gz).sum(-1) / torch.logical_or(z,gz).sum(-1)
                # x = x + gx
        else:
            x = obs

        net_out = self.net(x)
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
            # and look in appendix G. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi, {'mean': torch.tanh(mu), 'std': std}


class ActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, goal_key, G=None):
        super().__init__()
        act_dim = act_space.shape[0]

        self.preproc = None
        # build policy and value functions
        self.pi = SquashedGaussianActor(
            obs_space, act_dim, goal_key, G=G, preproc=self.preproc
        )
        self.q1 = QFunction(obs_space, act_dim, goal_key, G=G, preproc=self.preproc)
        self.q2 = QFunction(obs_space, act_dim, goal_key, G=G, preproc=self.preproc)
        if G.learned_alpha:
            self.target_entropy = -np.prod(act_space.shape)
            self.log_alpha = torch.nn.Parameter(-0.5 * torch.ones(1))
        self.G = G

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _, ainfo = self.pi(obs, deterministic, False)
            return a

    def value(self, obs, act):
        with torch.no_grad():
            q1 = self.q1(obs, act)
            q2 = self.q1(obs, act)
            return (q1 + q2) / 2
