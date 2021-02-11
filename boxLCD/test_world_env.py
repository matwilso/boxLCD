import time
from pyglet.gl import glClearColor
import argparse
import itertools
from collections import defaultdict
import copy
import sys, math
import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from boxLCD import utils
from boxLCD import Box, C, Dropbox
import pyglet
KEY = pyglet.window.key
A = utils.A

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  parser = argparse.ArgumentParser()
  for key, value in C.items():
    parser.add_argument(f'--{key}', type=utils.args_type(value), default=value)
  C = parser.parse_args()
  env = Box(C)
  #env = Dropbox(C)
  start = env.reset()
  env.render(mode='human')
  key_handler = KEY.KeyStateHandler()
  # monkey patch the env window to get keyboard input
  window = env.viewer.window
  window.push_handlers(key_handler)
  # set up variables
  paused = False
  past_keys = {}
  reset_on_done = False
  plotting = False
  obs_log = False
  omax = 0.0
  ret = 0

  # KEY BINDINGS
  # 0 - reset env
  # 1 - toggle automatic reset on done
  # SPACE - pause
  # P - plot
  # O - print obs
  # ESC - quit

  # RUN THE ENV AND RENDER IT
  for i in itertools.count(0):
    action = env.action_space.sample()
    action = np.zeros_like(action)
    daction = utils.NamedArray(action, env.act_info, do_map=False)
    curr_keys = defaultdict(lambda: False)
    curr_keys.update({key: val for key, val in key_handler.items()})
    check = lambda x: curr_keys[x] and not past_keys[x]

    if check(KEY._0) or check(KEY.NUM_0):
      start = env.reset()
      time.sleep(0.1)
    if check(KEY.SPACE):
      paused = not paused
    if check(KEY.P):
      plotting = not plotting
    if check(KEY.O):
      obs_log = not obs_log
    if check(KEY._1):
      reset_on_done = not reset_on_done
    if check(KEY.ESCAPE): 
      exit()

    if not paused or check(KEY.RIGHT):
      act = env.action_space.sample()
      obs, rew, done, info = env.step(act)
      nobs = utils.NamedArray(obs, env.obs_info, do_map=False)
      if obs_log:
        print(nobs.todict())
      ret += rew
      if done and reset_on_done:
        print('episode return',  ret)
        ret = 0
        start = obs = env.reset()
    img = env.render(mode='human')
    #def upsize(img):
    #  frac = 128//img.shape[0]
    #  return 255*img[...,None].astype(np.uint8).repeat(3,-1).repeat(frac, -2).repeat(frac,-3)
    #wh = A[8,8]
    #resolutions = [wh*2**i for i in [0,1,2,4]]
    #lcds = [upsize(env.lcd_render(*wh)) for wh in resolutions][::-1]
    #img = np.concatenate(lcds, 1)
    #plt.imsave(f'imgs/{i}.png', img)
    #if i == 200:
    #  break

    if plotting:
      plt.imshow(img); plt.show()
    past_keys = {key: val for key, val in curr_keys.items()}