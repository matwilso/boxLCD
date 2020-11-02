import numpy as np
import torch.nn as nn
from torch import distributions
import torch.nn.functional as F
import torch
from tensorflow import nest
import utils
from jax.tree_util import tree_multimap

class EnsembleDyn(nn.Module):
    def __init__(self, act_n, cfg):
        super().__init__()
        self.cfg = cfg
        self.models = nn.ModuleList([RSSM(act_n, cfg) for _ in range(cfg.num_ens)])

class RSSM(nn.Module):
    def __init__(self, act_n, cfg):
        super().__init__()
        self.cfg = cfg
        self.cell = nn.GRUCell(self.cfg.hidden, self.cfg.deter)
        self.obs1 = nn.Linear(1024+self.cfg.deter, self.cfg.hidden)
        self.obs2 = nn.Linear(self.cfg.hidden, 2*self.cfg.stoch)
        self.img1 = nn.Linear(self.cfg.stoch+act_n, self.cfg.hidden)
        self.img2 = nn.Linear(self.cfg.deter, self.cfg.hidden)
        self.img3 = nn.Linear(self.cfg.hidden, 2*self.cfg.stoch)

    def initial(self, bs):
        init = dict(mean=torch.zeros([bs, self.cfg.stoch]), std=torch.zeros([bs, self.cfg.stoch]), stoch=torch.zeros([bs, self.cfg.stoch]), deter=torch.zeros([bs, self.cfg.deter]))
        return nest.map_structure(lambda x: x.to(self.cfg.device), init)

    def observe(self, embed, action, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        embed = embed.permute([1, 0, 2])
        action = action.permute([1, 0, 2])
        post, prior = utils.static_scan(lambda prev, inputs: self.obs_step(prev[0], *inputs), (action, embed), (state, state))
        post = {k: v.permute([1, 0, 2]) for k, v in post.items()}
        prior = {k: v.permute([1, 0, 2]) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action.permute([1, 0, 2])
        prior = utils.static_scan(self.img_step, action, state)
        prior = {k: v.permute([1, 0, 2]) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        return torch.cat([state['stoch'], state['deter']], -1)

    def get_dist(self, state):
        return distributions.MultivariateNormal(state['mean'], scale_tril=torch.diag_embed(state['std']))

    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior['deter'], embed], -1)
        x = self.obs1(x)
        x = F.relu(x)
        x = self.obs2(x)
        mean, std = torch.split(x, self.cfg.stoch, -1)
        std = F.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).rsample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
        return post, prior

    def img_step(self, prev_state, prev_action):
        x = torch.cat([prev_state['stoch'], prev_action], -1)
        x = self.img1(x)
        x = F.relu(x)
        deter = self.cell(x, prev_state['deter'])
        x = self.img2(deter)
        x = F.relu(x)
        x = self.img3(x)
        mean, std = torch.split(x, self.cfg.stoch, -1)
        std = F.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).rsample()
        prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
        return prior

class ConvEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._depth = 32
        self.c1 = nn.Conv2d(3, self._depth, kernel_size=4, stride=2)
        self.c2 = nn.Conv2d(self._depth, 2*self._depth, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(2*self._depth, 4*self._depth, kernel_size=4, stride=2)
        self.c4 = nn.Conv2d(4*self._depth, 8*self._depth, kernel_size=4, stride=2)

    def forward(self, image):
        x = self.c1(image)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.c4(x)
        return x.flatten(1, -1)

class ConvDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._depth = 32
        self._shape = (64, 64, 3)
        self.l1 = nn.Linear(self.cfg.deter+self.cfg.stoch, 32*self._depth)
        self.d1 = nn.ConvTranspose2d(32*self._depth, 4*self._depth, 5, stride=2)
        self.d2 = nn.ConvTranspose2d(4*self._depth, 2*self._depth, 5, stride=2)
        self.d3 = nn.ConvTranspose2d(2*self._depth, 1*self._depth, 6, stride=2)
        self.d4 = nn.ConvTranspose2d(1*self._depth, 3, 6, stride=2)

    def forward(self, features):
        x = self.l1(features)
        x = x[..., None, None]
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.d3(x)
        x = F.relu(x)
        x = self.d4(x)
        #return distributions.Independent(distributions.Bernoulli(logits=x), len(self._shape))
        return distributions.Independent(distributions.Normal(x,1), len(self._shape))

class MLP(nn.Module):
    def __init__(self, sizes, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleDict()
        self.sizes = sizes
        for i in range(len(sizes)-1):
            self.layers[f'dense{i}'] = nn.Linear(sizes[i], sizes[i+1])

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.sizes)-1):
            x = self.layers[f'dense{i}'].forward(x)
            if i != (len(self.sizes)-2): x = F.relu(x)
        return x

class LatentFwds(nn.Module):
    def __init__(self, act_n, cfg):
        super().__init__()
        self.fwds = nn.ModuleList([MLP([act_n+cfg.stoch+cfg.deter, 128, 128, cfg.stoch+cfg.deter], cfg) for _ in range(cfg.num_ens)])

    def forward(self, obs):
        return torch.stack([f(obs) for f in self.fwds])

class DenseEncoder(nn.Module):
    def __init__(self, obs_n, cfg):
        super().__init__()
        self.mlp = MLP([obs_n, 128, 1024])

    def forward(self, obs):
        x = self.mlp.forward(obs)
        return x

class DenseDecoder(nn.Module):
    def __init__(self, out_n, cfg):
        super().__init__()
        self.mlp = MLP([cfg.stoch+cfg.deter, 128, out_n])

    def forward(self, features):
        x = self.mlp.forward(features)
        return distributions.Independent(distributions.Normal(x,1), 1)
        #if self._dist == 'normal':
        #    return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
        #if self._dist == 'binary':
        #    return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
        #raise NotImplementedError(self._dist)

MIN_STD = 1e-4
INIT_STD = 5
MEAN_SCALE = 5
class ActionDecoder(nn.Module):
    def __init__(self, act_n, cfg):
        super().__init__()
        self.cfg = cfg
        self.raw_init_std = torch.log(torch.exp(torch.tensor(INIT_STD).float()) - 1).to(cfg.device)
        self.mlp = MLP([cfg.stoch+cfg.deter, 128, act_n*2])
        self.act_n = act_n

    def forward(self, features):
        x = features
        x = self.mlp(x)
        mean, std = torch.split(x, self.act_n, -1)
        mean = MEAN_SCALE * torch.tanh(mean / MEAN_SCALE)
        std = F.softplus(std + self.raw_init_std) + MIN_STD
        dist = distributions.Normal(mean, std)
        dist = distributions.Independent(dist, 1)
        return dist