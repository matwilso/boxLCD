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
  C = parser.parse_args()
  env = envs.UrchinBall(C)
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
    out = env.render(mode='human')

    def proc(img):
      shape = 256 // img.shape[1]
      return 255 * img.astype(np.uint8)[..., None].repeat(shape, 0).repeat(shape, 1).repeat(3, 2)

    pretty = env.lcd_render(256, 128, pretty=True)
    lcd = proc(env.lcd_render(32, 16))

    idx = np.clip(0.9*(i-30), 0, 255).astype(np.int)
    if idx == 255:
      img = np.concatenate([lcd[:,:idx], np.zeros_like(lcd)[:,:2]], axis=1)
    else:
      img = np.concatenate([lcd[:,:idx], np.zeros_like(lcd)[:,:2], pretty[:,-(255-idx):]], axis=1)
    #idx = 255 - np.clip(2*(i - 50), 0, 255)
    #img = np.concatenate([pretty[:,:idx], np.zeros_like(lcd)[:,:2], lcd[:,-(256-idx):]], axis=1)
    #if i < 40:
    #  img = 255*env.lcd_render(256, 128, pretty=True)
    #elif i < 60:
    #  img = proc(env.lcd_render(256, 128))
    #elif i < 100:
    #  img = proc(env.lcd_render(128, 64))
    #elif i < 120:
    #  img = proc(env.lcd_render(64, 32))
    #elif i < 300:
    #  img = proc(env.lcd_render(32, 16))
    imgs += [img]

    #else:
    if i > 400:
      write_gif('test.gif', imgs, fps=30)
      break

    if KEY.ESCAPE in key_handler:
      exit()
