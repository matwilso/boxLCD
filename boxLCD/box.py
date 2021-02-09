import time
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
from envs.world_defs import World, Agent, Object
from envs.box2d import B2D
import pyglet
KEY = pyglet.window.key
A = utils.A

object_kwargs = dict(shape='circle', size=2.0, density=0.1)

class Dropbox(B2D):
  def __init__(self, C):
    w = World(agents=[], objects=[Object('object0', shape='box', size=2.0, density=0.1)])
    super().__init__(w, C)

class CrabObject(B2D):
  def __init__(self, C):
    w = World(agents=[Agent(f'{C.cname}0')], objects=[Object('object0', shape='random', size=2.0, density=0.1)])
    super().__init__(w, C)

class Box(B2D):
  def __init__(self, C):
    gravity = [0, -9.81]
    forcetorque = 0
    w = World(agents=[Agent(f'{C.cname}{i}') for i in range(C.num_agents)], objects=[Object(f'object{i}', **object_kwargs) for i in range(C.num_objects)], gravity=gravity, forcetorque=forcetorque)
    super().__init__(w, C)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from define_config import config, args_type
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  parser.set_defaults(**{'env_size': 320})
  C = parser.parse_args()
  env = Box(C)
  start = env.reset()
  #import ipdb; ipdb.set_trace()
  ret = 0
  env.render()
  key_handler = KEY.KeyStateHandler()
  env.viewer.window.push_handlers(key_handler)
  window = env.viewer.window

  @window.event
  def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    print()
    print(x,y,dx,dy, buttons, modifiers)
    cp = env.SCALE*env.dynbodies['crab0:root'].position
    if np.linalg.norm(A[cp] - A[x,y]) < 100:
      env.dynbodies['crab0:root'].position += (dx/env.SCALE,dy/env.SCALE)

  paused = False
  traj = []
  past_keys = {}
  delay = 0.1*1/16
  dor = False
  plotting = False
  omax = 0.0

  while True:
    action = env.action_space.sample()
    action = np.zeros_like(action)
    act_dict = env.get_act_dict(action)
    curr_keys = defaultdict(lambda: False)
    curr_keys.update({key: val for key, val in key_handler.items()})
    check = lambda x: curr_keys[x] and not past_keys[x]

    if check(KEY._0) or check(KEY.NUM_0):
      start = env.reset()
      time.sleep(0.1)
      traj = []
    if check(KEY.UP):
      act_dict['object0:force'] = 1.0
    if check(KEY.DOWN):
      act_dict['object0:force'] = -1.0
    if check(KEY.LEFT):
      act_dict['object0:theta'] = 0.5
    if check(KEY.RIGHT):
      act_dict['object0:theta'] = -0.5
    if check(KEY.SPACE):
      paused = not paused
    if check(KEY.P):
      plotting = not plotting
    if check(KEY._1):
      dor = not dor

    if check(KEY.S):
      delay *= 2.0 
    if check(KEY.F):
      delay *= 0.5
    time.sleep(delay)

    if check(KEY.NUM_4):
      pass
      # TODO: add support for rendering past images in traj

    if check(KEY.ESCAPE): 
      exit()

    if not paused or check(KEY.NUM_6):
      #obs, rew, done, info = env.step(np.zeros_like(env.action_space.sample()))
      act = env.action_space.sample()
      #print(act)
      #while True:
      #  env.render(action=act)
      #  if key_handler[KEY.RIGHT]: break
      obs, rew, done, info = env.step(act)
      dobs = utils.WrappedArray(obs, env.obs_info, map=False)
      #print(dobs['object0:x:p'])
      #omax = max(omax, np.max(dobs['luxo0:root:x:v', 'luxo0:root:y:v', 'luxo0:root:ang:v']))
      #print(omax)
      #print(np.linalg.norm(start[env.gixs] - obs[env.gixs], axis=-1) / len(env.gixs)**0.5)
      #print(env.get_obs_dict(obs))
      print(obs)
      #print()
      #print(dobs['object0:x:v', 'object0:y:v', 'object0:ang:v'])
      #print(dobs['luxo0:root:x:v', 'luxo0:root:y:v', 'luxo0:root:ang:v'])
      ret += rew
      #obs, rew, done, info = env.step(env.get_act_vec(act_dict))
      if done and dor:
        print(ret)
        ret = 0
        start = obs = env.reset()
      # print only the obs data that comes from object0
      #print(rew, utils.filter(env.get_obs_dict(obs, map=False), 'object0'))
      #print(obs.max(), env.obs_keys[obs.argmax()])
    bf = time.time()
    img = env.render()
    dr = time.time()-bf
    #print(dr)
    if plotting:
      plt.imshow(img); plt.show()
    past_keys = {key: val for key, val in curr_keys.items()}