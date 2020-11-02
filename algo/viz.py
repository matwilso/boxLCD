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
from torch.cuda import amp
from nets import models
import pyglet

class Viz(Trainer):
    def __init__(self, cfg, make_env):
        super().__init__(cfg, make_env)

    def run(self):
        self.refresh_dataset()
        batch = next(self.data_iter)
        #self.tenv.env.VIEWPORT_H *= 10
        #self.tenv.env.VIEWPORT_W *= 10
        self.tenv.reset()
        self.tenv.render()

        KEY = pyglet.window.key
        keys = KEY.KeyStateHandler()
        self.tenv.viewer.window.push_handlers(keys)
        window = self.tenv.viewer.window

        paused = False
        k = False
        l = False
        past_keys = {}

        ep_len = self.cfg.ep_len

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
            if check(KEY.K):
                k = not k
            if check(KEY.S):
                delay *= 2.0 
            if check(KEY.F):
                delay *= 0.5
            time.sleep(delay)

            if not paused:
                time_idx = time_idx + 1
            if time_idx > (ep_len-1) and k:
                batch_idx += 1
            time_idx = time_idx % ep_len


            if batch_idx > (ep_len-1):
                batch = next(self.data_iter)
                batch_idx = 0
                time_idx = 0

            state = np.array(batch['state'][batch_idx, time_idx])
            obs = utils.DWrap(state, self.tenv.env.obs_info)
            #obs['object0:x:p', 'object0:y:p'] *= 10
            self.tenv.env.visualize_obs(obs.arr)
            past_keys = {key: val for key, val in curr_keys.items()}