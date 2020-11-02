import atexit
import functools
import sys
import threading
import traceback

import gym
import numpy as np
from PIL import Image
from gym.utils import seeding, EzPickle

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
