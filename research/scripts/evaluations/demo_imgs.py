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
from boxLCD import G
from boxLCD import envs
import pyglet
KEY = pyglet.window.key
A = boxLCD.utils.A

def env_fn(G, seed=None):
  def _make():
    if G.env == 'dropbox':
      env = envs.Dropbox(G)
    elif G.env == 'bounce':
      env = envs.Bounce(G)
    elif G.env == 'boxor':
      env = envs.BoxOrCircle(G)
    elif G.env == 'urchin':
      env = envs.Urchin(G)
    elif G.env == 'urchin_ball':
      env = envs.UrchinBall(G)
    elif G.env == 'urchin_balls':
      env = envs.UrchinBalls(G)
    elif G.env == 'urchin_cubes':
      env = envs.UrchinCubes(G)
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
  for key, value in G.items():
    parser.add_argument(f'--{key}', type=boxLCD.utils.args_type(value), default=value)
  parser.add_argument(f'--env', type=str, default='urchin_ball')
  G = parser.parse_args()
  env = env_fn(G)()
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
    if i > env.G.ep_len:
      print(i)
      write_gif(f'{G.env}.gif', imgs, fps=env.G.fps)
      break

    if KEY.ESCAPE in key_handler:
      exit()
