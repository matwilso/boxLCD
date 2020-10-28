import time
from collections import defaultdict
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import nest
from torch import optim
import numpy as np
from algo.base import Trainer
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
        self.tenv.env.VIEWPORT_H *= 10
        self.tenv.env.VIEWPORT_W *= 10
        self.tenv.reset()
        self.tenv.render()

        KEY = pyglet.window.key
        keys = KEY.KeyStateHandler()
        self.tenv.viewer.window.push_handlers(keys)
        window = self.tenv.viewer.window

        paused = False
        k = False
        past_keys = {}

        batch_idx = 0
        time_idx = 0
        while True:
            curr_keys = defaultdict(lambda: False)
            curr_keys.update({key: val for key, val in keys.items()})
            if curr_keys[KEY.LEFT] and not past_keys[KEY.LEFT]:
                batch_idx -= 1
            if curr_keys[KEY.RIGHT] and not past_keys[KEY.RIGHT]:
                batch_idx += 1
            if curr_keys[KEY.UP] and not past_keys[KEY.UP]:
                time_idx -= 1
            if curr_keys[KEY.DOWN] and not past_keys[KEY.DOWN]:
                time_idx += 1
            if curr_keys[KEY.SPACE] and not past_keys[KEY.SPACE]:
                paused = not paused
            if curr_keys[KEY.ESCAPE]: 
                exit()
            if curr_keys[KEY.K] and not past_keys[KEY.K]:
                k = not k

            if not paused:
                time_idx = time_idx + 1
            if time_idx > 49 and not k:
                batch_idx += 1
                k = not k
            time_idx = time_idx % 50


            if batch_idx > 49:
                batch = next(self.data_iter)
                batch_idx = 0
                time_idx = 0


            state = np.array(batch['state'][batch_idx, time_idx])
            obs = utils.DWrap(state, self.tenv.env.obs_info)
            obs['object0:x:p', 'object0:y:p'] *= 10
            self.tenv.env.visualize_obs(obs.arr)
            past_keys = {key: val for key, val in curr_keys.items()}

            label = pyglet.text.Label('Hello, world', font_name='Times New Roman', font_size=36, x=window.width//2, y=window.height//2, anchor_x='center', anchor_y='center')
            label.draw()



