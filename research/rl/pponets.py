import numpy as np
import scipy.signal
import torch
import torch.distributions as thd
import torch.nn as nn

from research import utils


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
    def __init__(self, obs_space, act_dim, goal_key, G):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.goal_key = goal_key
        gsize = obs_space[goal_key].shape[-1]
        if True:
            size = obs_space[G.state_key].shape[-1] * 2
            self.goal_preproc = nn.Linear(gsize, size // 2)
        else:
            size = obs_space[G.state_key].shape[-1] + gsize
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
        if True:
            g = self.goal_preproc(obs[self.goal_key])
            x = torch.cat([obs[self.G.state_key], g], -1)
        else:
            x = torch.cat([obs[self.G.state_key], obs[self.goal_key]], -1)
        mu = self.net(x)
        std = torch.exp(self.log_std)
        return thd.Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, obs_space, act_dim, goal_key, G, preproc=None):
        super().__init__()
        self.G = G
        self.goal_key = goal_key
        gsize = obs_space[goal_key].shape[-1]
        if True:
            size = obs_space[G.state_key].shape[-1] * 2
            self.goal_preproc = nn.Linear(gsize, size // 2)
        else:
            size = obs_space[self.G.state_key].shape[-1] + gsize
        self.base = BaseMLP(size, 1, G)

    def forward(self, obs):
        if True:
            g = self.goal_preproc(obs[self.goal_key])
            x = torch.cat([obs[self.G.state_key], g], -1)
        else:
            x = torch.cat([obs[self.G.state_key], obs[self.goal_key]], -1)
        return self.base(x).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, goal_key, G=None):
        super().__init__()
        act_dim = act_space.shape[-1]
        self.G = G
        self.pi = Actor(obs_space, act_dim, goal_key, G)
        # build value function
        self.v = Critic(obs_space, act_dim, goal_key, G)

    def step(self, obs):
        obs = {
            key: torch.as_tensor(1.0 * val, dtype=torch.float32).to(self.G.device)
            for key, val in obs.items()
        }
        with torch.no_grad():
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

    def save(self, dir):
        print("SAVED PPO", dir)
        path = dir / f'ppo_ac.pt'
        sd = self.state_dict()
        sd['G'] = self.G
        torch.save(sd, path)
        print(path)

    def load(self, dir):
        path = dir / f'ppo_ac.pt'
        sd = torch.load(path, map_location=self.G.device)
        G = sd.pop('G')
        self.load_state_dict(sd)
        print(f'LOADED PPO {path}')
