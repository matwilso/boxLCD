import pathlib
import time
from collections import defaultdict
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import nest
from torch import optim
import numpy as np
from algo.trainer import Trainer
from torch import distributions
import utils
from nets import models
import pyglet
from data import records
from jax.tree_util import tree_multimap, tree_map

class Viz(Trainer):
    def __init__(self, cfg, make_env):
        super().__init__(cfg, make_env)

    def run(self):
        N = self.cfg.ep_len // self.cfg.bl
        files = list(sorted(map(lambda x: str(x), pathlib.Path(self.barrel_path).glob('*.tfrecord'))))[-2*N:-N]
        #files = list(sorted(map(lambda x: str(x), pathlib.Path(self.barrel_path).glob('*.tfrecord'))))[-N:]
        #files = list(sorted(map(lambda x: str(x), pathlib.Path(self.barrel_path).glob('*.tfrecord'))))
        #num_files = len(files)
        #idx = np.random.randint(0, num_files // N)
        #files = files[N*idx:N*idx+N]
        #self.data_iter = records.make_dataset(self.barrel_path, self.state_shape, self.image_shape, self.act_n, self.cfg, shuffle=False, files=files, repeat=False)

        batches = []
        for f in files:
            self.data_iter = records.make_dataset(self.barrel_path, self.state_shape, self.image_shape, self.act_n, self.cfg, shuffle=False, files=[f], repeat=False)
            batches += [next(self.data_iter)]
        batch = tree_multimap(lambda x,*y: np.stack([x,*y],1), batches[0], *batches[1:])
        batch = tree_map(lambda x: x.reshape([50, -1, x.shape[-1]]), batch)
        #self.tenv.env.VIEWPORT_H *= 10
        #self.tenv.env.VIEWPORT_W *= 10
        self.tenv.reset()
        self.tenv.render()

        # TODO: add an option here where we can view a contiguous rollout. like stitch together some tfrecords by loading them. like manually maybe
        # may need some dataset mods.

        KEY = pyglet.window.key
        keys = KEY.KeyStateHandler()
        self.tenv.viewer.window.push_handlers(keys)
        window = self.tenv.viewer.window

        paused = False
        k = False
        l = False
        past_keys = {}

        bl = N*self.cfg.bl

        batch_idx = 0
        time_idx = 0
        delay = 0.1*1/4
        while True:
            curr_keys = defaultdict(lambda: False)
            curr_keys.update({key: val for key, val in keys.items()})
            check = lambda x: curr_keys[x] and not past_keys[x]

            if check(KEY.LEFT):
                batch_idx -= 1
                time_idx = 0
            if check(KEY.RIGHT):
                batch_idx += 1
                time_idx = 0
            if check(KEY.UP):
                time_idx -= 1
            if check(KEY.DOWN):
                time_idx += 1
            if check(KEY.SPACE):
                paused = not paused
            if check(KEY.ESCAPE):
                exit()
            if check(KEY.I):
                import ipdb; ipdb.set_trace()
            if check(KEY.K):
                k = not k
            if check(KEY.S):
                delay *= 2.0 
            if check(KEY.F):
                delay *= 0.5
            time.sleep(delay)

            if not paused:
                time_idx = time_idx + 1
            if time_idx > (bl-1) and k:
                batch_idx += 1
            time_idx = time_idx % bl

            if batch_idx > (bl-1):
                batch = next(self.data_iter)
                batch_idx = 0
                time_idx = 0

            rew = np.mean(batch['rew'][batch_idx])
            state = np.array(batch['state'][batch_idx, time_idx])
            obs = utils.DWrap(state, self.tenv.env.obs_info)
            #obs['object0:x:p', 'object0:y:p'] *= 10
            self.tenv.env.visualize_obs(obs.arr, f'b {batch_idx} t{time_idx} r{rew:.2f}')
            past_keys = {key: val for key, val in curr_keys.items()}