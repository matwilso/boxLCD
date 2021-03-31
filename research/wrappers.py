import copy
import atexit
import functools
import sys
import threading
import traceback

import gym
import numpy as np
from PIL import Image
from gym.utils import seeding, EzPickle
from research import utils


class RewardGoalEnv:
  def __init__(self, env, C):
    self._env = env
    self.SCALE = 2
    self.C = C

  @property
  def action_space(self):
    return self._env.action_space

  @property
  def observation_space(self):
   base_space = self._env.observation_space
   base_space.spaces['goal:lcd'] = base_space.spaces['lcd']
   base_space.spaces['goal:pstate'] = base_space.spaces['pstate']
   return base_space

  def reset(self, *args, **kwargs):
    self.goal = self._env.reset()
    obs = self._env.reset(*args, **kwargs)
    #self.goal = obs = self._env.reset(*args, **kwargs)
    obs['goal:lcd'] = np.array(self.goal['lcd'])
    obs['goal:pstate'] = np.array(self.goal['pstate'])
    return obs

  def simi2rew(self, similarity):
    """map [0,1] --> [-1,0] in an exponential mapping. (if you get linearly closer to 1.0, you get exponentially closer to 0.0)"""
    assert similarity >= 0.0 and similarity <= 1.0
    # return np.exp(SCALE*similarity) / np.exp(SCALE*1)
    return -1 + similarity
    #return -1 + np.exp(self.SCALE * (similarity - 1))

  #def rew2simi(self, rew):
  #  """map [-1,0] --> [-1,0] in a log mapping."""
  #  assert rew >= -1.0 and rew <= 0.0
  #  return (np.log(rew + 1) / self.SCALE) + 1
  def render(self, *args, **kwargs):
    self._env.render(*args, **kwargs)

  def comp_rew_done(self, obs, info={}):
    done = False
    if self.C.state_rew:
      delta = ((obs['goal:pstate'] - obs['pstate'])**2)
      keys = utils.filtlist(self._env.obs_keys, '.*(x|y):p')
      idxs = [self._env.obs_keys.index(x) for x in keys]
      delta = delta[idxs].mean()
      rew = -delta**0.5
      info['simi'] = delta
      if delta < 0.010:
        rew = 0
        #done = False
        done = True
    else:
      similarity = (np.logical_and(obs['lcd'] == 0, obs['lcd'] == obs['goal:lcd']).mean() / (obs['lcd'] == 0).mean())
      rew = -1 + similarity
      info['simi'] = similarity
      if similarity > 0.70:
        rew = 0
        #done = False
        done = True
    return rew, done

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    obs['goal:lcd'] = np.array(self.goal['lcd'])
    obs['goal:pstate'] = np.array(self.goal['pstate'])

    rew, done = self.comp_rew_done(obs, info)

    #similarity = (obs['goal:lcd'] == obs['lcd']).mean()
    #rew = self.simi2rew(similarity)
    rew = rew * self.C.rew_scale
    return obs, rew, done, info

  def close(self):
      self._env.close()

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from boxLCD.envs import Luxo
  import utils
  from rl.sacnets import ActorCritic
  import torch as th
  import pathlib
  import time
  C = utils.AttrDict()
  C.state_rew = 1
  C.net = 'vae'
  C.nfilter = 128
  C.hidden_size = 128
  C.learned_alpha = 0
  C.alpha = 0.2
  C.device = 'cpu'
  C.lcd_h = 16
  C.lcd_w = 32
  C.wh_ratio = 2.0
  C.lr = 1e-3
  #C.weightdir = pathlib.Path('logs/vaes/x2_beta0.5_1e-3_bigger128_bs32/')
  #C.weightdir = pathlib.Path('logs/vaes/carb/x2_beta0.1_1e-3_bigger128_bs32/')
  C.weightdir = pathlib.Path('logs/vaes/x2_beta1.0_1e-3_bigger128_bs32/')
  #C.lcd_base = 32
  C.rew_scale = 1.0
  env = Luxo(C)
  env = RewardGoalEnv(env, C)
  print(env.observation_space, env.action_space)
  env.reset()
  ac = ActorCritic(env.observation_space, env.action_space, C=C).to(C.device)
  while True:
    env.render(mode='human')
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    o = {key: th.as_tensor(val[None].astype(np.float32), dtype=th.float32).to(C.device) for key, val in obs.items()}
    r = ac.comp_rew(o)
    print(rew, info['simi'], r)
    time.sleep(0.1)
    #plt.imshow(obs['lcd'] != obs['goal:lcd']); plt.show()
    #plt.imshow(np.c_[obs['lcd'], obs['goal:lcd']]); plt.show()
    if done:
      break

