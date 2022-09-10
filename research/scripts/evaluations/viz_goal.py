import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from jax.tree_util import tree_map, tree_multimap

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
    env = env_fn(G)()
    if 'Urchin' in G.env:
        env.seed(1)
    elif 'Luxo' in G.env:
        env.seed(15)
    # env.seed(7)
    MODE = 1
    if MODE == 0:
        imgs = []
        for i in range(8):
            obs = env.reset()
            lcd = 1.0 * obs['lcd'][..., None].repeat(3, -1)
            goal = 1.0 * obs['goal:lcd'][..., None].repeat(3, -1)
            goal[..., 0] = 1.0
            img = np.minimum(lcd, goal)
            imgs += [img]
            imgs += [np.zeros_like(img)[:, :1]]
        img = np.concatenate(imgs[:-1], 1).repeat(8, 0).repeat(8, 1)
        img = np.cat([img, np.ones_like(img)[::4]], 0)
        plt.imsave(f'{G.env}_goals.png', img)
    else:
        all_imgs = []
        for i in range(2):
            imgs = []
            for j in range(4):
                obs = env.reset()
                lcd = 1.0 * obs['lcd'][..., None].repeat(3, -1)
                goal = 1.0 * obs['goal:lcd'][..., None].repeat(3, -1)
                goal[..., 0] = 1.0
                img = np.minimum(lcd, goal)
                imgs += [img]
                imgs += [np.zeros_like(img)[:, :1]]
            img = np.concatenate(imgs[:-1], 1)
            all_imgs += [img]
            all_imgs += [np.zeros_like(img)[:1]]
        img = np.concatenate(all_imgs[:-1]).repeat(8, 0).repeat(8, 1)
        # img = np.cat([img, np.ones_like(img)[::4]], 0)
        plt.imsave(f'{G.env}_goals.png', img)
