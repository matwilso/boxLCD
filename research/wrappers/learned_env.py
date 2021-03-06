import os
from re import I
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
import research.utils
import gym
from boxLCD import env_map
from research import utils
from research.wrappers.body_goal import BodyGoalEnv
from research.wrappers.cube_goal import CubeGoalEnv

def outproc(img):
  return (255 * img[..., None].repeat(3, -1)).astype(np.uint8).repeat(8, 1).repeat(8, 2)

# TODO: add back this wrapper probably, since it makes things a bit cleaner.
# i was wrong about something when i removed it.
class LearnedEnv:
  def __init__(self, num_envs, model, G):
    self.num_envs = num_envs
    self.window_batch = None
    self.G = G
    self.model = model
    #self.real_env = env_fn(G)()
    self.real_env = model.env
    self.obs_keys = self.real_env._env.obs_keys
    self.pobs_keys = self.real_env._env.pobs_keys
    self.model.load(G.weightdir)
    self.action_space = gym.spaces.Box(-1, +1, (num_envs,) + model.action_space.shape, model.action_space.dtype)
    self.model.eval()
    for p in self.model.parameters():
      p.requires_grad = False

    def act_sample(): return 2.0 * th.rand(self.action_space.shape).to(G.device) - 1.0
    self.action_space.sample = act_sample
    spaces = {}
    self.keys = ['lcd', 'proprio']
    for key in self.keys:
      val = self.real_env.observation_space.spaces[key]
      spaces[key] = gym.spaces.Box(-1, +1, (num_envs,) + val.shape, dtype=val.dtype)
    spaces['zstate'] = gym.spaces.Box(-1, +1, (num_envs, self.model.z_size), dtype=val.dtype)
    self.observation_space = gym.spaces.Dict(spaces)

  def reset(self, *args, update_window_batch=True, **kwargs):
    with th.no_grad():
      prompts = [self.real_env.reset() for _ in range(self.num_envs)]
      prompts = tree_multimap(lambda x, *y: th.as_tensor(np.stack([x, *y])).to(self.G.device), prompts[0], *prompts[1:])
      window_batch = {key: th.zeros([self.model.G.window, *val.shape], dtype=th.float32).to(self.G.device) for key, val in self.observation_space.spaces.items()}
      window_batch['action'] = th.zeros([self.model.G.window, *self.action_space.shape]).to(self.G.device)
      window_batch = {key: val.transpose(0, 1) for key, val in window_batch.items()}
      for key in self.keys:
        window_batch[key][:, 0] = prompts[key]

      # TODO: do more than one step.
      if self.G.reset_prompt:
        # with th.no_grad():
        #  window_batch = self.model.onestep(window_batch, self.ptr, temp=self.G.lenv_temp)
        self.ptr = 1
      else:
        window_batch['action'] += 2.0 * th.rand(window_batch['action'].shape).to(self.G.device) - 1.0
        with th.no_grad():
          for self.ptr in range(10):
            window_batch = self.model.onestep(window_batch, self.ptr, temp=self.G.lenv_temp)
          window_batch = {key: th.cat([val[:, 5:], th.zeros_like(val)[:, :5]], 1) for key, val in window_batch.items()}
          self.ptr = 4

      obs = {key: val[:, self.ptr - 1] for key, val in window_batch.items() if key in self.keys}
      if update_window_batch:  # False if we want to preserve the simulator state
        self.window_batch = window_batch
      self.ep_t = 0
      return obs

  def step(self, act):
    with th.no_grad():
      self.ep_t += 1
      with th.no_grad():
        self.window_batch['action'][:, self.ptr - 1] = th.as_tensor(act).to(self.G.device)
        self.window_batch = self.model.onestep(self.window_batch, self.ptr, temp=self.G.lenv_temp)
        obs = {key: val[:, self.ptr] for key, val in self.window_batch.items() if key in self.keys}
        self.ptr = min(1 + self.ptr, self.model.G.window - 1)
        if self.ptr == self.model.G.window - 1:
          self.window_batch = {key: th.cat([val[:, 1:], th.zeros_like(val)[:, :1]], 1) for key, val in self.window_batch.items()}
          self.ptr -= 1
      rew, done = th.zeros(self.num_envs).to(self.G.device), th.zeros(self.num_envs).to(self.G.device)
      done[:] = self.ep_t >= self.G.ep_len
      return obs, rew, done, {'timeout': done.clone()}

class RewardLenv:
  def __init__(self, env):
    self.lenv = env
    self.SCALE = 2
    self.G = env.G
    #self.real_env = env_fn(self.G)()._env
    self.real_env = self.lenv.real_env
    self.pobs_keys = self.lenv.pobs_keys
    self.obs_keys = self.lenv.obs_keys
    self.goal = {key: th.zeros(space.shape).to(self.G.device).float() for key, space in self.observation_space.spaces.items() if 'goal' in key}
    if self.real_env.__class__.__name__ == 'CubeGoalEnv':
      if self.G.arbiterdir.name != '':
        arbiter_path = list(self.G.arbiterdir.glob('*.pt'))
        if len(arbiter_path) > 0:
          arbiter_path = arbiter_path[0]
        self.obj_loc = th.jit.load(str(arbiter_path))
        self.obj_loc.eval()
        print('LOADED OBJECT LOCALIZER')
      else:
        self.obj_loc = None

  @property
  def action_space(self):
    return self.lenv.action_space

  @property
  def observation_space(self):
    base_space = copy.deepcopy(self.lenv.observation_space)
    base_space.spaces['goal:lcd'] = copy.deepcopy(base_space.spaces['lcd'])
    base_space.spaces['goal:proprio'] = copy.deepcopy(base_space.spaces['proprio'])
    if 'Cube' in self.real_env.__class__.__name__:
      base_space.spaces['goal:object'] = copy.deepcopy(base_space.spaces['proprio'])
      base_space.spaces['goal:object'].shape = (self.lenv.num_envs, 2)
    return base_space

  def step(self, act, logger=defaultdict(lambda: [])):
    with th.no_grad():
      with utils.Timer(logger, 'lenv_step'):
        obs, rew, ep_done, info = self.lenv.step(act)
      obs['goal:proprio'] = self.goal['goal:proprio'].detach().clone()
      obs['goal:lcd'] = self.goal['goal:lcd'].detach().clone()
      if 'goal:object' in self.goal:
        obs['goal:object'] = self.goal['goal:object'].detach().clone()

      with utils.Timer(logger, 'comp_rew_done'):
        rew, goal_done = self.comp_rew_done(obs, info)
      success = th.logical_and(goal_done.bool(), ~ep_done.bool())
      rew[success] += 1.0
      done = th.logical_or(ep_done.bool(), goal_done.bool())
      with utils.Timer(logger, 'post_proc'):
        rew = rew * self.G.rew_scale
        if self.G.autoreset:
          if th.all(ep_done):
            with utils.Timer(logger, 'reset_env'):
              obs = self.reset()
          else:
            with utils.Timer(logger, 'reset_goals'):
              if th.any(goal_done):
                self._reset_goals(goal_done, logger)
      self.last_obs = tree_map(lambda x: x.detach().clone(), obs)
      return obs, rew, done, info

  def _reset_goals(self, mask, logger=defaultdict(lambda: [])):
    with th.no_grad():
      mask = mask.bool()
      if self.G.lenv_goals:
        #assert not self.G.reset_prompt, 'we dont want to use prompts for this because its slow'
        new_goal = utils.prefix_dict('goal:', self.lenv.reset(update_window_batch=False))
        new_goal = utils.prefix_dict('goal:', utils.filtdict(self.lenv.reset(update_window_batch=False), '(lcd|proprio|object)'))
      else:
        with utils.Timer(logger, 'set_goal'):
          new_goal = [utils.filtdict(self.real_env.reset(), 'goal:(lcd|proprio|object)') for _ in np.arange(self.lenv.num_envs)]
          new_goal = tree_multimap(lambda x, *y: th.as_tensor(np.stack([x, *y])).to(self.G.device).float(), new_goal[0], *new_goal[1:])

      with utils.Timer(logger, 'tileup'):
        def tileup(x, y):
          while x.ndim != y.ndim:
            y = y[..., None]
          return y
        self.goal = tree_multimap(lambda x, y: (x * tileup(x, mask) + y * ~tileup(y, mask)).detach(), new_goal, self.goal)

  def reset(self, *args, **kwargs):
    with th.no_grad():
      self._reset_goals(th.ones(self.lenv.num_envs, dtype=th.int32).to(self.G.device))
      obs = self.lenv.reset(*args, **kwargs)
      obs['goal:lcd'] = self.goal['goal:lcd'].detach().clone()
      obs['goal:proprio'] = self.goal['goal:proprio'].detach().clone()
      if 'goal:object' in self.goal:
        obs['goal:object'] = self.goal['goal:object'].detach().clone()
      self.last_obs = tree_map(lambda x: x.detach().clone(), obs)
      return obs

  def render(self, *args, **kwargs):
    self.lenv.render(*args, **kwargs)

  def comp_rew_done(self, obs, info={}):
    done = th.zeros(obs['lcd'].shape[0]).to(self.G.device)
    if 'BodyGoal' in self.real_env.__class__.__name__:
      keys = utils.filtlist(self.pobs_keys, '.*(x|y):p')
      idxs = [self.pobs_keys.index(x) for x in keys]
      delta = (obs['goal:proprio'][..., idxs] - obs['proprio'][..., idxs]).abs()
      delta = delta.mean(-1)
      rew = -delta
      info['delta'] = delta.detach()
      #rew[delta < 0.010] = 0
      done[delta < self.G.goal_thresh] = 1
      info['success'] = done.detach()
    elif self.real_env.__class__.__name__ == 'CubeGoalEnv':
      if self.obj_loc is None:
        import ipdb; ipdb.set_trace()
      else:
        obj = self.obj_loc(obs).detach()
        goal = self.obj_loc(utils.filtdict(obs, 'goal:', fkey=lambda x: x[5:])).detach()
        delta = (obj - goal).abs().mean(-1)
        if self.G.diff_delt:
          last_obj = self.obj_loc(self.last_obs).detach()
          last_delta = (last_obj - goal).abs().mean(-1)
          rew = -0.05 + 10 * (last_delta - delta)
        else:
          rew = -delta
        done[delta < self.G.goal_thresh] = 1
        info['delta'] = delta.detach()
    else:
      import ipdb; ipdb.set_trace()
    return rew.detach(), done.detach()

if __name__ == '__main__':
  from research.nets import net_map
  import argparse
  from boxLCD import env_map
  from research.define_config import config, args_type, env_fn
  from PIL import Image, ImageDraw, ImageFont
  from research.wrappers.async_vector_env import AsyncVectorEnv
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  tempC, _ = parser.parse_known_args()
  # grab defaults from the env
  Env = env_map[tempC.env]
  parser.set_defaults(**Env.ENV_DG)
  parser.set_defaults(**{'goals': 1})
  G, _ = parser.parse_known_args()
  G.lcd_w = int(G.wh_ratio * G.lcd_base)
  G.lcd_h = G.lcd_base
  G.imsize = G.lcd_w * G.lcd_h
  G.lenv_temp = 1.0
  G.reset_prompt = 1
  G.diff_delt = 1
  G.goal_thresh = 0.01
  G.lenv_goals = 0

  env = env_fn(G)()

  sd = th.load(G.weightdir / f'{G.model}.pt')
  mG = sd.pop('G')
  mG.device = G.device
  model = net_map[G.model](env, mG)
  model.load(G.weightdir)
  model.to(G.device)
  G.num_vars = utils.count_vars(model)
  model.eval()
  for p in model.parameters():
    p.requires_grad = False
  print('LOADED MODEL', G.weightdir)
  lenv = RewardLenv(LearnedEnv(G.num_envs, model, G))
  obs = lenv.reset()
  start = time.time()
  lcds = [obs['lcd']]
  glcds = [obs['goal:lcd']]
  pslcds = [env.reset(proprio=obs['proprio'][0].cpu())['lcd']]
  rews = [np.zeros(8)]
  deltas = [np.zeros(8)]
  for i in range(200):
    act = lenv.action_space.sample()
    obs, rew, done, info = lenv.step(act)
    lcds += [obs['lcd']]
    glcds += [obs['goal:lcd']]
    pslcds += [env.reset(proprio=obs['proprio'][0].cpu())['lcd']]
    rews += [rew.detach().cpu().numpy()]
    deltas += [info['delta'].cpu().numpy()]

  lcds = th.stack(lcds).flatten(1, 2).cpu().numpy()
  glcds = th.stack(glcds).flatten(1, 2).cpu().numpy()
  lcds = (lcds - glcds + 1.0) / 2.0
  print('1', time.time() - start)
  lcds = outproc(lcds)

  dframes = []
  white = (255, 255, 255)
  for i in range(len(lcds)):
    frame = lcds[i]
    pframe = Image.fromarray(frame.astype(np.uint8))
    # get a drawing context
    draw = ImageDraw.Draw(pframe)
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 60)
    for j in range(8):
      draw.text((10, G.lcd_h * 8 * j + 10), f't: {i} r:{rews[i][j]:.3f}', fill=white, fnt=fnt)
    dframes += [np.array(pframe)]
  dframes = np.stack(dframes)
  # for i in range(G.num_envs):
  #  lcds[50:, 128*i:128*i+10] = [1, 0, 0]
  utils.write_gif('test.gif', dframes, fps=G.fps)
  #utils.write_gif('pstest.gif', outproc(np.stack(pslcds)), fps=G.fps)

  #obs = env.reset()
  #ostart = time.time()
  # for i in range(200):
  #  act = env.action_space.sample()
  #  obs, rew, done, info = env.step(act)
  #print('2', time.time() - ostart)

  #vstart = time.time()
  #venv = AsyncVectorEnv([env_fn(G) for _ in range(G.num_envs)])
  #obs = venv.reset(np.arange(G.num_envs))
  #lcds = [obs['lcd']]
  #glcds = [obs['goal:lcd']]
  # for i in range(200):
  #  act = venv.action_space.sample()
  #  obs, rew, done, info = venv.step(act)
  #  lcds += [obs['lcd']]
  #  glcds += [obs['goal:lcd']]
  # venv.close()
  #print('3', time.time() - vstart)
  #lcds = th.as_tensor(np.stack(lcds)).flatten(1, 2).cpu().numpy()
  #glcds = th.as_tensor(np.stack(glcds)).flatten(1, 2).cpu().numpy()
  #lcds = (1.0*lcds - 1.0*glcds + 1.0) / 2.0
  #print('1', time.time() - start)
  #utils.write_gif('realtest.gif', outproc(lcds), fps=G.fps)
