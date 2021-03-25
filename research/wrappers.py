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


class RewardGoalEnv:
  def __init__(self, env, C):
    self._env = env
    self.SCALE = 2
    self.C = C

  def __getattr__(self, name):
    return getattr(self._env, name)

  # @property
  # def action_space(self):
  #  return self.env.action_space

  @property
  def observation_space(self):
   base_space = self._env.observation_space
   base_space.spaces['goal:lcd'] = base_space.spaces['lcd']
   base_space.spaces['goal:pstate'] = base_space.spaces['pstate']
   return base_space

  def reset(self, *args, **kwargs):
    self.goal = self._env.reset()
    obs = self._env.reset(*args, **kwargs)
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

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    obs['goal:lcd'] = np.array(self.goal['lcd'])
    obs['goal:pstate'] = np.array(self.goal['pstate'])

    if self.C.state_rew:
      delta = ((obs['goal:pstate'] - obs['pstate'])**2)
      keys = utils.filtlist(self._env.obs_keys, '.*(x|y):p')
      idxs = [self._env.obs_keys.index(x) for x in keys]
      delta = delta[idxs].mean()
      rew = -delta**0.5
      info['simi'] = delta
      if delta < 0.010:
        done = True
    else:
      similarity = (np.logical_and(obs['lcd'] == 0, obs['lcd'] == obs['goal:lcd']).mean() / (obs['lcd'] == 0).mean())
      rew = -1 + similarity
      info['simi'] = similarity
      if similarity > 0.9:
        done = True
    #similarity = (obs['goal:lcd'] == obs['lcd']).mean()
    #rew = self.simi2rew(similarity)
    return obs, rew, done, info

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from boxLCD.envs import Luxo
  import utils
  C = utils.AttrDict()
  C.state_rew = 1
  env = Luxo()
  env = RewardGoalEnv(env, C)
  print(env.observation_space, env.action_space)
  env.reset()
  while True:
    env.render(mode='human')
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    print(rew, info['simi'])
    plt.imshow(obs['lcd'] != obs['goal:lcd']); plt.show()
    #plt.imshow(np.c_[obs['lcd'], obs['goal:lcd']]); plt.show()
    if done:
      break

