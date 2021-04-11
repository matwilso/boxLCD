import copy
import gym
import numpy as np
from gym.utils import seeding, EzPickle
from research import utils

class LearnedPreprocEnv:
  """
  Learned model that preprocesses observations and produces a `zstate`
  """
  def __init__(self, model, env, C, device='cpu'):
    self._model = model
    self._env = env
    self.SCALE = 2
    self.C = C
    self.device = device
    self.goal_based = self._env.goal_based

  @property
  def action_space(self):
    return self._env.action_space

  @property
  def observation_space(self):
    base_space = self._env.observation_space
    base_space.spaces['zstate'] = gym.spaces.Box(-10, 10, (self._model.z_size,))
    ## TODO: deal with goal_based envs. probably add a little indicator.
    #base_space.spaces['goal:lcd'] = copy.deepcopy(base_space.spaces['lcd'])
    return base_space

  def _preproc_obs(self, obs):
    import ipdb; ipdb.set_trace()

  def reset(self, *args, **kwargs):
    obs = self._env.reset(*args, **kwargs)
    return self._preproc_obs(obs)

  def render(self, *args, **kwargs):
    self._env.render(*args, **kwargs)

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    return self._preproc_obs(obs), rew, done, info

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
  C = utils.AttrDict()
  C.env = 'UrchinCube'
  C.state_rew = 1
  C.device = 'cpu'
  C.lcd_h = 16
  C.lcd_w = 32
  C.wh_ratio = 1.5
  C.lr = 1e-3
  #C.lcd_base = 32
  C.rew_scale = 1.0
  C.diff_delt = 1 
  C.fps = 10
  env = envs.UrchinCube(C)
  C.fps = env.C.fps
  env = CubeGoalEnv(env, C)
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
    #o = {key: th.as_tensor(val[None].astype(np.float32), dtype=th.float32).to(C.device) for key, val in obs.items()}
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
  utils.write_video('mtest.mp4', dframes, fps=C.fps)
  #utils.write_gif('test.gif', dframes, fps=C.fps)