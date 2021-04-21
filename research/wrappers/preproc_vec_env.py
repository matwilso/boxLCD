import torch as th
import copy
from re import I
import gym
import numpy as np
from gym.utils import seeding, EzPickle
from research import utils
from .async_vector_env import AsyncVectorEnv
from scipy.spatial.distance import cosine

class PreprocVecEnv:
  """
  Learned model that preprocesses observations and produces a `zstate`
  """
  def __init__(self, model, env, G, device='cuda'):
    self.model = model
    self._env = env
    self.SCALE = 2
    self.G = G
    self.device = device
    self.model.to(device)
    self.model.eval()
    self.shared_memory = env.shared_memory

  @property
  def action_space(self):
    return self._env.action_space

  @property
  def observation_space(self):
    import ipdb; ipdb.set_trace()
    base_space = self._env.observation_space
    #base_space.spaces['zstate'] = gym.spaces.Box(-1, 1, (self.model.z_size,))
    #if 'goal:proprio' in base_space.spaces:
    #  base_space.spaces['goal:zstate'] = gym.spaces.Box(-1, 1, (self.model.z_size,))
    return base_space

  def _preproc_obs(self, obs):
    batch_obs = {key: th.as_tensor(1.0 * val).float().to(self.device) for key, val in obs.items()}
    zstate = self.model.encode(batch_obs)
    obs['zstate'] = zstate.detach().cpu().numpy()
    #if 'goal:proprio' in batch_obs:
    #  goal = utils.filtdict(batch_obs, 'goal:', fkey=lambda x: x[5:])
    #  zgoal = self.model.encode(goal)
    #  #obs['goal:zstate'] = zgoal.detach().cpu().numpy()
    return obs

  def reset(self, *args, **kwargs):
    obs = self._env.reset(*args, **kwargs)
    return self._preproc_obs(obs)

  def render(self, *args, **kwargs):
    self._env.render(*args, **kwargs)

  def comp_rew(self, z, gz):
    cos = np.zeros(z.shape[0])
    for i in range(len(z)):
      cos[i] = -cosine(z[i],gz[i])
    return cos

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    obs = self._preproc_obs(obs)
    if self.G.preproc_rew:
      rew = self.comp_rew(obs['zstate'], obs['goal:zstate'])
    return obs, rew, done, info

  def close(self):
    self._env.close()

if __name__ == '__main__':
  from body_goal import BodyGoalEnv
  from PIL import Image, ImageDraw, ImageFont
  import matplotlib.pyplot as plt
  from boxLCD import envs
  import utils
  import torch as th
  import pathlib
  import time
  #from research.nets.bvae import BVAE
  from boxLCD import envs, env_map
  G = utils.AttrDict()
  G.env = 'Urchin'
  G.state_rew = 1
  G.device = 'cpu'
  G.lcd_h = 16
  G.lcd_w = 32
  G.wh_ratio = 2.0
  G.lr = 1e-3
  #G.lcd_base = 32
  G.rew_scale = 1.0
  G.diff_delt = 1 
  G.fps = 10
  G.hidden_size = 128
  G.nfilter = 128
  G.vqK = 128
  G.vqD = 128
  G.goal_thresh = 0.01
  env = envs.Urchin(G)
  G.fps = env.G.fps
  #model = BVAE(env, G)
  def env_fn(G, seed=None):
    def _make():
      env = envs.Urchin(G)
      env = BodyGoalEnv(env, G)
      return env
    return _make

  def outproc(img):
    return (255 * img[..., None].repeat(3, -1)).astype(np.uint8).repeat(8, 1).repeat(8, 2)

  start = time.time()
  env = AsyncVectorEnv([env_fn(G) for _ in range(8)])
  #env = PreprocVecEnv(model, env, G, device='cpu')
  obs = env.reset(np.arange(8))
  lcds = [obs['lcd']]
  glcds = [obs['goal:lcd']]
  for i in range(200):
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    lcds += [obs['lcd']]
    glcds += [obs['goal:lcd']]
  env.close()
  lcds = th.as_tensor(np.stack(lcds)).flatten(1, 2).cpu().numpy()
  glcds = th.as_tensor(np.stack(glcds)).flatten(1, 2).cpu().numpy()
  lcds = (1.0*lcds - 1.0*glcds + 1.0) / 2.0
  print('dt', time.time() - start)
  utils.write_gif('realtest.gif', outproc(lcds), fps=G.fps)
