import os
import time
from pyglet.gl import glClearColor
import argparse
import itertools
from collections import defaultdict
import copy
import sys
import math
import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import boxLCD.utils
from boxLCD import C
from boxLCD import envs
import pyglet
KEY = pyglet.window.key
A = boxLCD.utils.A

def env_fn(C, seed=None):
  def _make():
    if C.env == 'dropbox':
      env = envs.Dropbox(C)
    elif C.env == 'bounce':
      env = envs.Bounce(C)
    elif C.env == 'boxor':
      env = envs.BoxOrCircle(C)
    elif C.env == 'urchin':
      env = envs.Urchin(C)
    elif C.env == 'urchin_ball':
      env = envs.UrchinBall(C)
    elif C.env == 'urchin_balls':
      env = envs.UrchinBalls(C)
    elif C.env == 'urchin_cubes':
      env = envs.UrchinCubes(C)
    env.seed(seed)
    return env
  return _make


def write_gif(name, frames, fps=30):
  start = time.time()
  from moviepy.editor import ImageSequenceClip
  # make the moviepy clip
  clip = ImageSequenceClip(list(frames), fps=fps)
  clip.write_gif(name, fps=fps)
  print(time.time() - start)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  parser = argparse.ArgumentParser()
  for key, value in C.items():
    parser.add_argument(f'--{key}', type=boxLCD.utils.args_type(value), default=value)
  parser.add_argument(f'--env', type=str, default='urchin_ball')
  C = parser.parse_args()
  env = env_fn(C)()
  env.seed(7)
  np_random = np.random.RandomState(4)
  start = env.reset()
  env.render(mode='human')
  key_handler = KEY.KeyStateHandler()
  window = env.viewer.window
  window.push_handlers(key_handler)
  MAX = 256

  imgs = []
  for i in itertools.count(0):
    action = np_random.uniform(-1,1,env.action_space.shape[0])
    env.step(action)
    out = env.render(mode='human', return_pyglet_view=True)

    def proc(img):
      shape = 256 // img.shape[1]
      return 255 * img.astype(np.uint8)[..., None].repeat(shape, 0).repeat(shape, 1).repeat(3, 2)
    imgs += [out]

    #else:
    if i > env.C.ep_len:
      print(i)
      write_gif(f'{C.env}.gif', imgs, fps=env.C.fps)
      break

    if KEY.ESCAPE in key_handler:
      exit()
