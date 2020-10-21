import atexit
import functools
import sys
import threading
import traceback

import gym
import numpy as np
from PIL import Image
from envs.base import BaseIndexEnv
from gym.utils import seeding, EzPickle


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
