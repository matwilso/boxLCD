import os
from jax.tree_util import tree_multimap, tree_map
import time
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
import torch as th
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
from datetime import datetime
from boxLCD.utils import A, NamedArray
import utils
import gym
from boxLCD import env_map

def outproc(img):
  return (255 * img[..., None].repeat(3, -1)).astype(np.uint8).repeat(8, 1).repeat(8, 2)

class RewardLenv:
  def __init__(self, env):
    self.lenv = env
    self.SCALE = 2
    self.C = env.C
    #self.real_env = env_fn(self.C)()._env
    self.real_env = self.lenv.real_env
    self.pobs_keys = self.lenv.pobs_keys
    self.obs_keys = self.lenv.obs_keys
    self.goal = {key: th.zeros(space.shape).to(self.C.device).float() for key, space in self.observation_space.spaces.items() if 'goal' in key}

  @property
  def action_space(self):
    return self.lenv.action_space

  @property
  def observation_space(self):
    base_space = self.lenv.observation_space
    base_space.spaces['goal:lcd'] = base_space.spaces['lcd']
    base_space.spaces['goal:pstate'] = base_space.spaces['pstate']
    return base_space

  def step(self, act):
    obs, rew, done, info = self.lenv.step(act)
    obs['goal:pstate'] = self.goal['goal:pstate'].detach().clone()
    obs['goal:lcd'] = self.goal['goal:lcd'].detach().clone()
    rew, _done = self.comp_rew_done(obs, info)
    done = th.logical_or(done, _done)
    rew = rew * self.C.rew_scale
    return obs, rew, done, info

  def _reset_goals(self, mask):
    mask = mask.bool()
    if not self.C.lenv_goals:
      new_goal = [utils.filtdict(self.real_env.reset(), 'goal:') for _ in np.arange(self.lenv.num_envs)]
      new_goal = tree_multimap(lambda x, *y: th.as_tensor(np.stack([x, *y])).to(self.C.device).float(), new_goal[0], *new_goal[1:])
    else:
      #assert not self.C.reset_prompt, 'we dont want to use prompts for this because its slow'
      new_goal = utils.prefix_dict('goal:', self.lenv.reset(update_window_batch=False))
    def tileup(x, y):
      while x.ndim != y.ndim:
        y = y[...,None]
      return y
    self.goal = tree_multimap(lambda x, y: x*tileup(x,mask) + y*~tileup(y,mask), new_goal, self.goal)

  def reset(self, *args, **kwargs):
    self._reset_goals(th.ones(self.lenv.num_envs, dtype=th.int32).to(self.C.device))
    obs = self.lenv.reset(*args, **kwargs)
    obs['goal:lcd'] = self.goal['goal:lcd'].detach().clone()
    obs['goal:pstate'] = self.goal['goal:pstate'].detach().clone()
    return obs

  def render(self, *args, **kwargs):
    self.lenv.render(*args, **kwargs)

  def comp_rew_done(self, obs, info={}):
    done = th.zeros(obs['lcd'].shape[0]).to(self.C.device)
    if self.C.state_rew:
      delta = ((obs['goal:pstate'] - obs['pstate'])**2)
      keys = utils.filtlist(self.pobs_keys, '.*(x|y):p')
      idxs = [self.pobs_keys.index(x) for x in keys]
      delta = delta[..., idxs].mean(-1)
      rew = -delta**0.5
      info['simi'] = delta
      #rew[delta < 0.010] = 0
      done[delta < self.C.goal_thresh] = 1
      info['success'] = done
    else:
      import ipdb; ipdb.set_trace()
      similarity = (np.logical_and(obs['lcd'] == 0, obs['lcd'] == obs['goal:lcd']).mean() / (obs['lcd'] == 0).mean())
      rew = -1 + similarity
      info['simi'] = similarity
      if similarity > 0.70:
        rew = 0
        #done = False
        done = True
    return rew, done


# TODO: add back this wrapper probably, since it makes things a bit cleaner.
# i was wrong about something when i removed it.
class LearnedEnv:
  def __init__(self, num_envs, model, C):
    self.num_envs = num_envs
    self.window_batch = None
    self.C = C
    self.model = model
    self.real_env = env_fn(C)()
    self.obs_keys = self.real_env._env.obs_keys
    self.pobs_keys = self.real_env._env.pobs_keys
    self.model.load(C.weightdir)
    self.action_space = gym.spaces.Box(-1, +1, (num_envs,) + model.action_space.shape, model.action_space.dtype)
    self.model.eval()
    for p in self.model.parameters():
      p.requires_grad = False

    def act_sample():
      return 2.0 * th.rand(self.action_space.shape).to(C.device) - 1.0
    self.action_space.sample = act_sample

    spaces = {}
    self.keys = ['lcd', 'pstate']
    for key in self.keys:
      val = self.real_env.observation_space.spaces[key]
      spaces[key] = gym.spaces.Box(-1, +1, (num_envs,) + val.shape, dtype=val.dtype)
    self.observation_space = gym.spaces.Dict(spaces)

  def reset(self, *args, update_window_batch=True, **kwargs):
    prompts = [self.real_env.reset() for _ in range(self.num_envs)]
    prompts = tree_multimap(lambda x, *y: th.as_tensor(np.stack([x, *y])).to(self.C.device), prompts[0], *prompts[1:])
    window_batch = {key: th.zeros([self.C.window, *val.shape], dtype=th.float32).to(self.C.device) for key, val in self.observation_space.spaces.items()}
    window_batch['acts'] = th.zeros([self.C.window, *self.action_space.shape]).to(self.C.device)
    window_batch = {key: val.transpose(0, 1) for key, val in window_batch.items()}
    for key in self.keys:
      window_batch[key][:, 0] = prompts[key]
    if self.C.reset_prompt:
      #with th.no_grad():
      #  window_batch = self.model.onestep(window_batch, self.ptr, temp=self.C.lenv_temp)
      self.ptr = 1
    else:
      window_batch['acts'] += 2.0 * th.rand(window_batch['acts'].shape).to(self.C.device) - 1.0
      with th.no_grad():
       for self.ptr in range(10):
         window_batch = self.model.onestep(window_batch, self.ptr, temp=self.C.lenv_temp)
       window_batch = {key: th.cat([val[:, 5:], th.zeros_like(val)[:, :5]], 1) for key, val in window_batch.items()}
       self.ptr = 4
      
    obs = {key: val[:, self.ptr-1] for key, val in window_batch.items() if key in self.keys}
    if update_window_batch: # False if we want to preserve the simulator state
      self.window_batch = window_batch
    return obs

  def step(self, act):
    with th.no_grad():
      self.window_batch['acts'][:, self.ptr - 1] = th.as_tensor(act).to(self.C.device)
      self.window_batch = self.model.onestep(self.window_batch, self.ptr, temp=self.C.lenv_temp)
      obs = {key: val[:, self.ptr] for key, val in self.window_batch.items() if key in self.keys}
      self.ptr = min(1 + self.ptr, self.C.window - 1)
      if self.ptr == self.C.window - 1:
        self.window_batch = {key: th.cat([val[:, 1:], th.zeros_like(val)[:, :1]], 1) for key, val in self.window_batch.items()}
        self.ptr -= 1
    rew, done = th.zeros(self.num_envs).to(self.C.device), th.zeros(self.num_envs).to(self.C.device)
    return obs, rew, done, {}


if __name__ == '__main__':
  from research.define_config import config, args_type, env_fn
  from research.nets.flat_everything import FlatEverything
  from research import utils
  import argparse
  from boxLCD import env_map
  from wrappers import AsyncVectorEnv
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  tempC, _ = parser.parse_known_args()
  # grab defaults from the env
  Env = env_map[tempC.env]
  parser.set_defaults(**Env.ENV_DC)
  parser.set_defaults(**{'goals': 1})
  C, _ = parser.parse_known_args()
  C.lcd_w = int(C.wh_ratio * C.lcd_base)
  C.lcd_h = C.lcd_base
  C.imsize = C.lcd_w * C.lcd_h
  C.lenv_temp = 2.0
  C.reset_prompt = 0
  C.diff_delt = 0
  C.goal_thresh = 0.01

  env = env_fn(C)()

  MC = th.load(C.weightdir / 'flatev2.pt').pop('C')
  model = FlatEverything(env, MC)
  lenv = RewardLenv(LearnedEnv(C.num_envs, model, C))
  obs = lenv.reset()
  start = time.time()
  lcds = [obs['lcd']]
  glcds = [obs['goal:lcd']]
  pslcds = [env.reset(pstate=obs['pstate'][0].cpu())['lcd']]
  for i in range(200):
    act = lenv.action_space.sample()
    obs, rew, done, info = lenv.step(act)
    lcds += [obs['lcd']]
    glcds += [obs['goal:lcd']]
    pslcds += [env.reset(pstate=obs['pstate'][0].cpu())['lcd']]

  lcds = th.stack(lcds).flatten(1, 2).cpu().numpy()
  glcds = th.stack(glcds).flatten(1, 2).cpu().numpy()
  #lcds = (lcds - glcds + 1.0) / 2.0
  print('1', time.time() - start)
  utils.write_gif('test.gif', outproc(lcds), fps=C.fps)
  utils.write_gif('pstest.gif', outproc(np.stack(pslcds)), fps=C.fps)

  obs = env.reset()
  ostart = time.time()
  for i in range(200):
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
  print('2', time.time() - ostart)

  vstart = time.time()
  venv = AsyncVectorEnv([env_fn(C) for _ in range(C.num_envs)])
  obs = venv.reset(np.arange(C.num_envs))
  lcds = [obs['lcd']]
  glcds = [obs['goal:lcd']]
  for i in range(200):
    act = venv.action_space.sample()
    obs, rew, done, info = venv.step(act)
    lcds += [obs['lcd']]
    glcds += [obs['goal:lcd']]
  venv.close()
  print('3', time.time() - vstart)
  lcds = th.as_tensor(np.stack(lcds)).flatten(1, 2).cpu().numpy()
  glcds = th.as_tensor(np.stack(glcds)).flatten(1, 2).cpu().numpy()
  lcds = (1.0*lcds - 1.0*glcds + 1.0) / 2.0
  print('1', time.time() - start)
  utils.write_gif('realtest.gif', outproc(lcds), fps=C.fps)

