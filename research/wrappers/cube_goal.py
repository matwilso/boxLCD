import copy
import gym
import numpy as np
from gym.utils import seeding, EzPickle
from research import utils

class CubeGoalEnv:
  def __init__(self, env, G):
    self._env = env
    self.SCALE = 2
    self.G = G
    self.keys = utils.filtlist(self._env.obs_keys, 'object.*(x|y):p')
    self.idxs = [self._env.obs_keys.index(x) for x in self.keys]
    self.rootkeys = utils.filtlist(self._env.obs_keys, '.*root.*(x|y):p')
    self.root_idxs = [self._env.obs_keys.index(x) for x in self.rootkeys]
    #self.obs_keys = self._env.obs_keys
    #self.obs_idxs = self._env.obs_idxs

  @property
  def action_space(self):
    return self._env.action_space

  @property
  def observation_space(self):
    base_space = self._env.observation_space
    base_space.spaces['goal:lcd'] = copy.deepcopy(base_space.spaces['lcd'])
    base_space.spaces['goal:proprio'] = copy.deepcopy(base_space.spaces['proprio'])
    base_space.spaces['goal:object'] = copy.deepcopy(base_space.spaces['proprio'])
    base_space.spaces['goal:object'].shape = (2,)
    base_space.spaces['goal:full_state'] = copy.deepcopy(base_space.spaces['full_state'])
    return base_space

  def reset(self, *args, **kwargs):
    self.goal = self._env.reset()
    for i in range(10):
      self.goal = self._env.step(np.zeros(self._env.action_space.shape))[0]
    obs = self._env.reset(*args, **kwargs)
    #self.goal = obs = self._env.reset(*args, **kwargs)
    obs['goal:lcd'] = np.array(self.goal['lcd'])
    obs['goal:full_state'] = np.array(self.goal['full_state'])
    obs['goal:proprio'] = np.array(self.goal['proprio'])
    obs['goal:object'] = np.array(self.goal['full_state'][..., self.idxs])
    self.last_obs = copy.deepcopy(obs)
    return obs

  def render(self, *args, **kwargs):
    self._env.render(*args, **kwargs)

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    obs['goal:lcd'] = np.array(self.goal['lcd'])
    obs['goal:full_state'] = np.array(self.goal['full_state'])
    obs['goal:proprio'] = np.array(self.goal['proprio'])
    obs['goal:object'] = np.array(self.goal['full_state'][..., self.idxs])
    rew, _done = self.comp_rew_done(obs, info)
    done = done or _done
    #similarity = (obs['goal:lcd'] == obs['lcd']).mean()
    #rew = self.simi2rew(similarity)
    rew = rew * self.G.rew_scale
    self.last_obs = copy.deepcopy(obs)
    return obs, rew, done, info

  def comp_rew_done(self, obs, info={}):
    done = False
    delta = ((obs['goal:full_state'][...,self.idxs] - obs['full_state'][..., self.idxs])**2).mean()
    if self.G.diff_delt:
      last_delta = ((obs['goal:full_state'][...,self.idxs] - self.last_obs['full_state'][..., self.idxs])**2).mean()
      # rew = 1*(last_delta**0.5 - delta**0.5) # reward should be proportional to how much closer we got.
      # rew = -0.1 + 5*(last_delta**0.5 - delta**0.5) # reward should be proportional to how much closer we got.
      #print(last_delta**0.5 - delta**0.5)
      rew = -0.05 + 10 * (last_delta**0.5 - delta**0.5)
    else:
      rew = -delta**0.5
    #rew = -1.0 + 0.5*movement
    #info['delta'] = delta
    done = False
    if delta < 0.01:
      done = True
      rew += 1.0
      #info['success'] = True
    # if delta < 0.005:
    # done = False
    return rew, done

  def close(self):
    self._env.close()

if __name__ == '__main__':
  from PIL import Image, ImageDraw, ImageFont
  import matplotlib.pyplot as plt
  from boxLCD import envs
  import utils
  from rl.sacnets import ActorCritic
  import torch as th
  import pathlib
  import time
  G = utils.AttrDict()
  G.env = 'UrchinCube'
  G.state_rew = 1
  G.device = 'cpu'
  G.lcd_h = 16
  G.lcd_w = 32
  G.wh_ratio = 1.5
  G.lr = 1e-3
  #G.lcd_base = 32
  G.rew_scale = 1.0
  G.diff_delt = 1 
  G.fps = 10
  env = envs.UrchinCube(G)
  G.fps = env.G.fps
  env = CubeGoalEnv(env, G)
  print(env.observation_space, env.action_space)
  obs = env.reset()
  lcds = [obs['lcd']]
  glcds = [obs['goal:lcd']]
  rews = [-np.inf]
  deltas = [-np.inf]
  while True:
    env.render(mode='human')
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    #o = {key: th.as_tensor(val[None].astype(np.float32), dtype=th.float32).to(G.device) for key, val in obs.items()}
    lcds += [obs['lcd']]
    glcds += [obs['goal:lcd']]
    rews += [rew]
    deltas += [info['delta']]
    #plt.imshow(obs['lcd'] != obs['goal:lcd']); plt.show()
    #plt.imshow(np.c_[obs['lcd'], obs['goal:lcd']]); plt.show()
    if done:
      break

  def outproc(img):
    return (255 * img[..., None].repeat(3, -1)).astype(np.uint8).repeat(8, 1).repeat(8, 2)
  lcds = np.stack(lcds)
  glcds = np.stack(glcds)
  lcds = (1.0 * lcds - 1.0 * glcds + 1.0) / 2.0
  lcds = outproc(lcds)
  dframes = []
  for i in range(len(lcds)):
    frame = lcds[i]
    pframe = Image.fromarray(frame)
    # get a drawing context
    draw = ImageDraw.Draw(pframe)
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 60)
    color = (255, 255, 255)
    draw.text((10, 10), f't: {i} r: {rews[i]:.3f}\nd: {deltas[i]:.3f}', fill=color, fnt=fnt)
    dframes += [np.array(pframe)]
  dframes = np.stack(dframes)
  utils.write_video('mtest.mp4', dframes, fps=G.fps)
  #utils.write_gif('test.gif', dframes, fps=G.fps)