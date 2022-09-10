import itertools
import pickle

import ignite
import matplotlib.pyplot as plt
import numpy as np
import torch
from jax.tree_util import tree_map, tree_multimap

from boxLCD.utils import A
from research import data, utils
from research.define_config import env_fn
from research.nets import net_map
from research.wrappers import AsyncVectorEnv


class Fiddler:
    def __init__(self, model, env, G):
        super().__init__()
        self.env = env
        self.xkeys = utils.filtlist(self.env._env.obs_keys, 'urchin.*x:p')
        self.xidxs = [self.env._env.obs_keys.index(x) for x in self.xkeys]
        sd = torch.load(G.weightdir / f'{G.model}.pt')
        mG = sd.pop('G')
        mG.device = G.device
        model = net_map[G.model](env, mG)
        model.load(G.weightdir)
        model.to(G.device)
        model.eval()
        self.G = G
        self.model = model

    def run(self):
        all_obses = []
        allz = []

        for i in range(100):
            obs = self.env.reset()
            cache = np.array(obs['full_state'][self.env.idxs[:1]])
            obses = [obs]
            obses2 = [obs]
            for j in range(100):
                # _o = self.env.reset()
                # _o['full_state'][self.env.idxs] = cache
                # obs = self.env.reset(full_state=_o['full_state'])
                obs['full_state'][self.env.idxs[:1]] = np.random.uniform(-1, 1)
                obs = self.env.reset(full_state=obs['full_state'])
                self.env.render(mode='human')
                obses += [obs]
            # all_obses += [obses]
            # print(i)
            obses = tree_multimap(lambda x, *y: np.stack([x, *y]), obses[0], *obses[1:])
            # obses = tree_map(lambda x: np.stack([*x], 0), obses)
            obses = tree_map(
                lambda v: torch.as_tensor(v, dtype=torch.float32).to(self.G.device), obses
            )
            out = self.model.encode(obses, noise=False, quantize=False)
            # print(out.var(0).topk(16)[1])
            # allz += [out.var(0).topk(16)[1]]

            for j in range(100):
                # _o = self.env.reset()
                # _o['full_state'][self.env.idxs] = cache
                # obs = self.env.reset(full_state=_o['full_state'])
                obs['full_state'][self.xidxs] += np.random.uniform(-0.1, 0.1)

                # obs['full_state'][self.env.idxs[:1]] = np.random.uniform(-1,1)
                obs = self.env.reset(full_state=obs['full_state'])
                self.env.render(mode='human')
                obses2 += [obs]
            # all_obses += [obses]
            print(i)
            obses2 = tree_multimap(
                lambda x, *y: np.stack([x, *y]), obses2[0], *obses2[1:]
            )
            # obses2 = tree_map(lambda x: np.stack([*x], 0), obses2)
            obses2 = tree_map(
                lambda v: torch.as_tensor(v, dtype=torch.float32).to(self.G.device), obses2
            )
            out2 = self.model.encode(obses2, noise=False, quantize=False)
            # print(out.var(0).topk(16)[1])
            allz += [(out.var(0) - out2.var(0)).topk(16)[1]]
        x = torch.bincount(torch.stack(allz).flatten(), minlength=self.model.z_size)
        weights = x / x.max()
        print(weights)
        with open('vec_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)
        y = (x > 3).nonzero()
        print(torch.cat([y, x[y]], -1))
