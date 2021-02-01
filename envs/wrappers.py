import atexit
import functools
import sys
import threading
import traceback

import gym
import numpy as np
from PIL import Image
from gym.utils import seeding, EzPickle
import utils

class NormalEnv(gym.Env, EzPickle):
    def __init__(self, env):
        self.env = env

    @property
    def observation_space(self):
        spaces = {}
        spaces['state'] = self.env.observation_space
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def viewer(self):
        return self.env.viewer

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return {'state': obs}, rew, done, info

    def reset(self):
        obs = self.env.reset()
        return {'state': obs}

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)

class PixelEnv(gym.Env, EzPickle):
    def __init__(self, env):
        self.env = env

    @property
    def observation_space(self):
        spaces = {}
        spaces['state'] = self.env.observation_space
        spaces['image'] = gym.spaces.Box(0, 255, (64, 64, 3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return {'state': obs, 'image': self.env.render()}, rew, done, info

    def reset(self):
        obs = self.env.reset()
        return {'state': obs, 'image': self.env.render()}

class LCDEnv(gym.Env, EzPickle):
    def __init__(self, env):
        self.env = env
        self.C = env.C

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        spaces = {}
        partial_obs_keys = utils.nlfilter(self.env.obs_keys, 'object')
        self.num_pobs = len(partial_obs_keys)
        self.pobs_idxs = [self.env.obs_keys.index(x) for k in partial_obs_keys]
        if self.num_pobs == 0:
            spaces['state'] = gym.spaces.Box(-1, +1, (1,), dtype=np.float32)
        else:
            spaces['state'] = gym.spaces.Box(-1, +1, (self.num_pobs,), dtype=np.float32)
        spaces['lcd'] = gym.spaces.Box(0, 1, (self.C.lcd_h, self.C.lcd_w), dtype=np.bool)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        state, rew, done, info = self.env.step(action)
        state = state[self.pobs_idxs] if self.num_pobs != 0 else 0.0
        return {'state': state, 'lcd': self.env.lcd_render()}, rew, done, info

    def lcd_render(self):
        return self.env.lcd_render()

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        state = state[self.pobs_idxs] if self.num_pobs != 0 else 0.0
        return {'state': state, 'lcd': self.env.lcd_render()}

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)