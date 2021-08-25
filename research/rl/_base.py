from re import I
import numpy as np
from research import wrappers
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from research.define_config import env_fn
import gym
from gym.vector.async_vector_env import AsyncVectorEnv
import torch as th
from research.nets import net_map
from jax.tree_util import tree_multimap, tree_map
from research import utils
import PIL
from PIL import Image, ImageDraw, ImageFont
TN = 8

class RLAlgo:
  def __init__(self, G):
    self.G = G
    print(G.full_cmd)
    # th.manual_seed(G.seed)
    # np.random.seed(G.seed)
    # Set up logger and save configuration
    self.logger = defaultdict(lambda: [])
    self.writer = SummaryWriter(G.logdir)
    self.tenv = env_fn(G, G.seed)()  # test env
    self.obs_space = self.tenv.observation_space
    self.act_space = self.tenv.action_space
    self.real_tvenv = AsyncVectorEnv([env_fn(G) for _ in range(TN)])
    if G.lenv:
      sd = th.load(G.weightdir / f'{G.model}.pt')
      mG = sd.pop('G')
      mG.device = G.device
      model = net_map[G.model](self.tenv, mG)
      model.to(G.device)
      model.eval()
      for p in model.parameters():
        p.requires_grad = False
      if G.model == 'FRNLD':
        Lenv = wrappers.TransformerLenv
      elif G.model == 'RSSM':
        Lenv = wrappers.RSSMLenv
      else:
        import ipdb; ipdb.set_trace()

      self.env = wrappers.RewardLenv(Lenv(G.num_envs, model, G))
      self.tvenv = self.learned_tvenv = wrappers.RewardLenv(Lenv(TN, model, G))
      #self.obs_space.spaces = utils.subdict(self.obs_space.spaces, self.env.observation_space.spaces.keys())
      self.obs_space = self.env.observation_space

      def fx(x):
        x.shape = x.shape[1:]
        return x
      self.obs_space.spaces = tree_map(fx, self.env.observation_space.spaces)
      if G.preproc:
        preproc = model.encoder
        #preproc = model.ronald
        self.env = wrappers.PreprocVecEnv(preproc, self.env, G)
        self.tvenv = self.learned_tvenv = wrappers.PreprocVecEnv(preproc, self.learned_tvenv, G)
        self.real_tvenv = wrappers.PreprocVecEnv(preproc, self.real_tvenv, G)
        self.obs_space.spaces['zstate'] = gym.spaces.Box(-1, 1, (preproc.z_size,))
        if 'goal:proprio' in self.obs_space.spaces:
          self.obs_space.spaces['goal:zstate'] = gym.spaces.Box(-1, 1, (preproc.z_size,))
    else:
      self.env = AsyncVectorEnv([env_fn(G) for _ in range(G.num_envs)])
      self.tvenv = self.real_tvenv
      if G.preproc:
        sd = th.load(G.weightdir / f'{G.model}.pt')
        mG = sd.pop('G')
        mG.device = G.device
        preproc = net_map[G.model](self.tenv, mG)
        preproc.to(G.device)
        preproc.load(G.weightdir)
        for p in preproc.parameters():
          p.requires_grad = False
        preproc.eval()
        self.env = wrappers.PreprocVecEnv(preproc, self.env, G)
        self.real_tvenv = self.tvenv = wrappers.PreprocVecEnv(preproc, self.tvenv, G)
        self.obs_space.spaces['zstate'] = gym.spaces.Box(-1, 1, (preproc.z_size,))
        if 'goal:proprio' in self.obs_space.spaces:
          self.obs_space.spaces['goal:zstate'] = gym.spaces.Box(-1, 1, (preproc.z_size,))
    # tenv.reset()
    if self.tenv.__class__.__name__ == 'BodyGoalEnv':
      self.goal_key = 'goal:proprio'
    elif self.tenv.__class__.__name__ == 'CubeGoalEnv':
      self.goal_key = 'goal:object'

  def get_av(self, o):
    raise NotImplementedError()

  def test_agent(self, itr, use_lenv=False):
    # init
    REP = 4
    if use_lenv:
      pf = th
      _env = self.learned_tvenv
      o, ep_ret, ep_len = _env.reset(), th.zeros(TN).to(self.G.device), th.zeros(TN).to(self.G.device)
    else:
      pf = np
      _env = self.real_tvenv
      o, ep_ret, ep_len = _env.reset(), np.zeros(TN), np.zeros(TN)

    # run
    frames = []
    dones = []
    rs = []
    vs = []
    all_done = pf.zeros_like(ep_ret)
    success = pf.zeros_like(ep_ret)
    for i in range(self.G.ep_len):
      # Take deterministic actions at test time
      a, v = self.get_av(o)
      #a, v, logp = self.ac.step(o)
      if not use_lenv and self.G.lenv:
        a = a.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
      o, r, d, info = _env.step(a)
      all_done = pf.logical_or(all_done, d)
      if i != (self.G.ep_len - 1):
        success = pf.logical_or(success, d)
      rs += [r]
      vs += [v]
      dones += [d]
      ep_ret += r * ~all_done
      ep_len += 1 * ~all_done
      if 'lcd' in o:
        delta = (1.0 * o['lcd'] - 1.0 * o['goal:lcd'] + 1) / 2
        #frame = np.concatenate([1.0 * o['goal:lcd'], 1.0 * o['lcd'], delta], axis=-2)
        frame = delta
        frames += [frame]
      else:
        frames = []

    if use_lenv:
      def proc(x): return x.detach().cpu().float()
      prefix = 'learned'
    else:
      def proc(x): return x
      prefix = 'real'
    if len(frames) != 0:
      if use_lenv:
        frames = th.stack(frames)
        frames = frames.detach().cpu().numpy()
      else:
        frames = np.stack(frames)
      frames = frames[..., None].repeat(REP, -3).repeat(REP, -2).repeat(3, -1)
      frames = frames.transpose(0, 2, 1, 3, 4).reshape([-1, self.G.lcd_h * 1 * REP, TN * self.G.lcd_w * REP, 3])
      # make borders
      for k in range(TN):
        if use_lenv:
          frames[:, :, k * REP * self.G.lcd_w] = [0, 0, 1]
        else:
          frames[:, :, k * REP * self.G.lcd_w] = [1, 0, 0]
        #frames[:, :, k * REP * self.G.lcd_w] = 0.0

      dframes = []
      yellow = (255, 255, 50)
      white = (255, 255, 255)
      purple = (75, 0, 130)
      for i in range(len(frames)):
        frame = frames[i]
        pframe = Image.fromarray((frame * 255).astype(np.uint8))
        # get a drawing context
        draw = ImageDraw.Draw(pframe)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 60)
        for j in range(TN):
          if use_lenv:
            color = yellow if dones[i][j].detach().cpu().numpy() and i != self.G.ep_len - 1 else white
            draw.text((self.G.lcd_w * REP * j + 10, 10), f't: {i} r:{rs[i][j].detach().cpu().numpy():.3f}\nV: {vs[i][j].detach().cpu().numpy():.3f}', fill=color, fnt=fnt)
            draw.text((self.G.lcd_w * REP * j + 5, 5), f'{"*"*int(success[j].detach().cpu().numpy())}', fill=yellow, fnt=fnt)
            #draw.text((self.G.lcd_w * REP * (j+1) - 20, 10), '[]', fill=purple, fnt=fnt)
          else:
            color = yellow if dones[i][j] and i != self.G.ep_len - 1 else white
            draw.text((self.G.lcd_w * REP * j + 10, 10), f't: {i} r:{rs[i][j]:.3f}\nV: {vs[i][j]:.3f}', fill=color, fnt=fnt)
            draw.text((self.G.lcd_w * REP * j + 5, 5), f'{"*"*int(success[j])}', fill=yellow, fnt=fnt)
        dframes += [np.array(pframe)]
      dframes = np.stack(dframes)
      vid = dframes.transpose(0, -1, 1, 2)[None]
      utils.add_video(self.writer, f'{prefix}_rollout', vid, itr + 1, fps=self.G.fps)
      print('wrote video', prefix)
    self.logger[f'{prefix}_test/EpRet'] += [proc(ep_ret).mean()]
    self.logger[f'{prefix}_test/EpLen'] += [proc(ep_len).mean()]
    self.logger[f'{prefix}_test/success_rate'] += [proc(success).mean()]
