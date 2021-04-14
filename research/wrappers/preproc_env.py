import torch as th
import pathlib
import copy
from re import I
import gym
import numpy as np
from gym.utils import seeding, EzPickle
from research import utils
from scipy.spatial.distance import cosine
from research.nets.bvae import BVAE
from research.nets.vae import VAE

class PreprocEnv:
  """
  Learned model that preprocesses observations and produces a `zstate`
  """
  def __init__(self, env, C, device='cpu'):
    #weightdir = pathlib.Path('./logs/qvae/binary_vqd32_save5_64_again')
    #MC = th.load(weightdir / 'bvae.pt').pop('C')
    #self.model = BVAE(env, MC)
    #self.model.load(weightdir)

    weightdir = pathlib.Path('./logs/qvae/urchincube/justvae_64_2/')
    MC = th.load(weightdir / 'vae.pt').pop('C')
    self.model = VAE(MC)
    self.model.load(weightdir)

    self._env = env
    self.SCALE = 2
    self.C = C
    self.device = device
    self.model.to(device)
    self.model.eval()
    self.e2 = envs.UrchinCube(C)
    self.e2 = CubeGoalEnv(self.e2, C)

  @property
  def action_space(self):
    return self._env.action_space

  @property
  def observation_space(self):
    base_space = self._env.observation_space
    base_space.spaces['zstate'] = gym.spaces.Box(-1, 1, (self.model.z_size,))
    if 'goal:pstate' in base_space.spaces:
      base_space.spaces['goal:zstate'] = gym.spaces.Box(-1, 1, (self.model.z_size,))
    return base_space

  def _preproc_obs(self, obs):
    batch_obs = {key: th.as_tensor(1.0 * val[None]).float().to(self.C.device) for key, val in obs.items()}
    zstate = self.model.encode(batch_obs)
    obs['zstate'] = zstate.detach().cpu().numpy()
    if 'goal:pstate' in batch_obs:
      goal = utils.filtdict(batch_obs, 'goal:', fkey=lambda x: x[5:])
      zgoal = self.model.encode(goal)
      obs['goal:zstate'] = zgoal.detach().cpu().numpy()
    return obs

  def reset(self, *args, **kwargs):
    obs = self._env.reset(*args, **kwargs)
    return self._preproc_obs(obs)

  def render(self, *args, **kwargs):
    self._env.render(*args, **kwargs)

  def comp_rew(self, z, gz):
    return -cosine(z[0],gz[0])
    import ipdb; ipdb.set_trace()
    simi = np.logical_and(z,gz).sum(-1) /  np.logical_or(z,gz).sum(-1)
    rew = np.exp(self.SCALE * (simi - 1))
    #x = 1- simi
    return -1+rew[0]

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    obs = self._preproc_obs(obs)
    #full_state = copy.deepcopy(obs['full_state'])
    #full_state[self._env.idxs] = 100
    #o2 = self.e2.reset(full_state=full_state)
    #gstate = copy.deepcopy(obs['goal:full_state'])
    #gstate[self._env.idxs] = 100
    #g2 = self.e2.reset(full_state=gstate)
    #o2['goal:lcd'] = g2['lcd']
    #o2 = self._preproc_obs(o2)
    #z = obs['zstate'] - o2['zstate']
    #gz = obs['goal:zstate'] - o2['goal:zstate']
    ##rew = self.comp_rew(obs['zstate'], obs['goal:zstate'])
    #rew = self.comp_rew(z, gz)
    return obs, rew, done, info

  def close(self):
    self._env.close()

if __name__ == '__main__':
  from body_goal import BodyGoalEnv
  from cube_goal import CubeGoalEnv
  from PIL import Image, ImageDraw, ImageFont
  import matplotlib.pyplot as plt
  from boxLCD import envs
  import utils
  from rl.sacnets import ActorCritic
  import torch as th
  import pathlib
  import time
  from research.nets.bvae import BVAE
  C = utils.AttrDict()
  C.env = 'UrchinCube'
  #C.env = 'Luxo'
  C.state_rew = 1
  C.device = 'cpu'
  C.lcd_h = 16
  C.lcd_w = 32
  C.wh_ratio = 2.0
  C.lr = 1e-3
  #C.lcd_base = 32
  C.rew_scale = 1.0
  C.diff_delt = 1 
  C.fps = 10
  C.hidden_size = 128
  C.nfilter = 128
  C.vqK = 128
  C.vqD = 128
  C.goal_thresh = 0.01
  #env = envs.Luxo(C)
  #weightdir = pathlib.Path('./logs/qvae/binary_vqd32_save5_64_again')
  #MC = th.load(weightdir / 'bvae.pt').pop('C')
  #model = BVAE(env, MC)
  #model.load(weightdir)
  env = envs.UrchinCube(C)
  C.fps = env.C.fps
  env = CubeGoalEnv(env, C)
  #env = BodyGoalEnv(env, C)
  env = PreprocEnv(env, C, device='cpu')
  print(env.observation_space, env.action_space)
  start = time.time()
  obs = env.reset()
  lcds = [obs['lcd']]
  glcds = [obs['goal:lcd']]
  rews = [-np.inf]
  deltas = [-np.inf]
  for i in range(200):
  #while True:
    #env.render(mode='human')
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    #o = {key: th.as_tensor(val[None].astype(np.float32), dtype=th.float32).to(C.device) for key, val in obs.items()}
    lcds += [obs['lcd']]
    glcds += [obs['goal:lcd']]
    rews += [rew]
    deltas += [info['delta']]
    #plt.imshow(obs['lcd'] != obs['goal:lcd']); plt.show()
    #plt.imshow(np.c_[obs['lcd'], obs['goal:lcd']]); plt.show()
    #if done:
    #  break

  def outproc(img):
    return (255 * img[..., None].repeat(3, -1)).astype(np.uint8).repeat(8, 1).repeat(8, 2)
  lcds = np.stack(lcds)
  glcds = np.stack(glcds)
  lcds = (1.0 * lcds - 1.0 * glcds + 1.0) / 2.0
  lcds = outproc(lcds)
  print('dt', time.time()-start)
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
  utils.write_video('preproc.mp4', dframes, fps=C.fps)
  #utils.write_gif('test.gif', dframes, fps=C.fps)