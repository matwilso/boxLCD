import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from jax.tree_util import tree_map, tree_map

from boxLCD import env_map
from research import data, runners, utils
from research.define_config import args_type, config, env_fn
from research.nets import net_map

np.cat = np.concatenate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    for key, value in config().items():
        parser.add_argument(f'--{key}', type=args_type(value), default=value)
    temp_cfg = parser.parse_args()
    # grab defaults from the env
    Env = env_map[temp_cfg.env]
    parser.set_defaults(**Env.ENV_DG)
    G = parser.parse_args()
    G.device = 'cpu'
    env = env_fn(G)()
    device = torch.device(G.device)
    sd = torch.load(G.weightdir / f'{G.model}.pt', map_location=device)
    mG = sd.pop('G')
    mG.device = G.device
    model = net_map[G.model](env, mG)
    model.load(G.weightdir)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print('LOADED MODEL', G.weightdir)
    # model.to(G.device)
    env.seed(7)
    # env.seed(1)
    # env.seed(7)
    np_random = np.random.RandomState(5)
    # TODO: burn in the env for a few steps first.
    obses = tree_map(lambda x: torch.as_tensor(x).float()[None, None], env.reset())
    actions = []
    for i in range(mG.window - 1):
        # for i in range(mG.ep_len - 1):
        action = np_random.uniform(-1, 1, env.action_space.shape[0])
        actions += [action]
        obs = env.step(action)[0]
        obses = tree_map(
            lambda x, y: torch.cat([x, torch.as_tensor(y).float()[None, None]], 1),
            obses,
            obs,
        )
    action = np_random.uniform(-1, 1, env.action_space.shape[0])
    actions += [action]
    action = torch.as_tensor(np.stack(actions)).float()[None]
    obses['action'] = action
    PN = 1
    sample = model.sample(1, action=action, prompts=obses, prompt_n=PN)
    # obses = tree_map(lambda x: x.cpu().numpy(), obses)
    imgs = []
    for i in range(0, 20, 1):
        t = obses['lcd'][0, i].numpy()
        x = sample['lcd'][0, i, 0].numpy()
        error = (t - x + 1.0) / 2.0
        blank = np.zeros_like(x)[:1]
        xt = np.cat([t, blank, x, blank, error], 0)[..., None].repeat(3, -1)
        imgs += [xt]
        col_blank = np.zeros_like(xt)[:, :1]
        if i == PN - 1:
            col_blank[..., 0] = 1.0
        imgs += [col_blank]

    img = np.cat(imgs[:-1], 1).repeat(8, 0).repeat(8, 1)
    img = np.cat([img, np.ones_like(img)[::4]], 0)
    plt.imsave(f'{mG.env}_frames.png', img)
