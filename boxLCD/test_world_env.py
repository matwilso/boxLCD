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
from boxLCD import C
from boxLCD import envs
import pyglet
KEY = pyglet.window.key
A = utils.A

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  parser = argparse.ArgumentParser()
  for key, value in C.items():
    parser.add_argument(f'--{key}', type=utils.args_type(value), default=value)
  C = parser.parse_args()
  #env = envs.Dropbox(C)
  #env = envs.Bounce(C)
  #env = envs.Urchin(C)
  env = envs.UrchinBall(C)
  #env = envs.UrchinBalls(C)
  #env = envs.UrchinCubes(C)
  #env = envs.BoxOrCircle(C)
  start = env.reset()
  env.render(mode='human')
  key_handler = KEY.KeyStateHandler()
  # monkey patch the env window to get keyboard input
  window = env.viewer.window
  window.push_handlers(key_handler)
  # set up variables
  paused = False
  past_keys = {}
  reset_on_done = True
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
    #action = np.zeros_like(action)
    curr_keys = defaultdict(lambda: False)
    curr_keys.update({key: val for key, val in key_handler.items()})
    check = lambda x: curr_keys[x] and not past_keys[x]

    if check(KEY._0) or check(KEY.NUM_0):
      #env.seed(0)
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
      action = env.action_space.sample()
      #action = np.zeros_like(action)
      obs, rew, done, info = env.step(action)
      nobs = utils.NamedArray(obs, env.obs_info, do_map=False)
      if obs_log:
        print(nobs.todict())
      ret += rew
      if done and reset_on_done:
        print('episode return',  ret)
        ret = 0
        #env.seed(0)
        start = obs = env.reset()
    img = env.render(mode='human')

    if plotting:
      plt.imshow(img); plt.show()
    past_keys = {key: val for key, val in curr_keys.items()}