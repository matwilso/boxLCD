from functools import WRAPPER_UPDATES, update_wrapper
from re import I
from research.nets.autoencoders.bvae import BVAE
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import PIL
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from copy import deepcopy
import itertools
import numpy as np
import torch as th
from torch.optim import Adam
import gym
import time
import numpy as np
import scipy.signal
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from research.rl.buffers import OGRB, ReplayBuffer, PPOBuffer
from research.rl.pponets import ActorCritic
from research.define_config import config, args_type, env_fn
from boxLCD import env_map
import boxLCD
from research import utils
from research import wrappers
#from research.nets.flat_everything import FlatEverything
from jax.tree_util import tree_multimap, tree_map
from research.nets import net_map
from gym.vector.async_vector_env import AsyncVectorEnv


def ppo(G):
  print(G.full_cmd)
  # th.manual_seed(G.seed)
  # np.random.seed(G.seed)

  # Set up logger and save configuration
  logger = defaultdict(lambda: [])
  writer = SummaryWriter(G.logdir)
  tenv = env_fn(G, G.seed)()  # test env
  obs_space = tenv.observation_space
  act_space = tenv.action_space
  TN = 8
  real_tvenv = AsyncVectorEnv([env_fn(G) for _ in range(TN)])
  if G.lenv:
    sd = th.load(G.weightdir / f'{G.model}.pt')
    mG = sd.pop('G')
    mG.device = G.device
    model = net_map[G.model](tenv, mG)
    model.to(G.device)
    model.eval()
    for p in model.parameters():
      p.requires_grad = False
    env = wrappers.RewardLenv(wrappers.LearnedEnv(G.num_envs, model, G))
    tvenv = learned_tvenv = wrappers.RewardLenv(wrappers.LearnedEnv(TN, model, G))
    #obs_space.spaces = utils.subdict(obs_space.spaces, env.observation_space.spaces.keys())
    obs_space = env.observation_space
    def fx(x):
      x.shape = x.shape[1:]
      return x
    obs_space.spaces = tree_map(fx, env.observation_space.spaces)

    if G.preproc:
      preproc = model.ronald
      env = wrappers.PreprocVecEnv(preproc, env, G)
      tvenv = learned_tvenv = wrappers.PreprocVecEnv(preproc, learned_tvenv, G)
      real_tvenv = wrappers.PreprocVecEnv(preproc, real_tvenv, G)
      obs_space.spaces['zstate'] = gym.spaces.Box(-1, 1, (preproc.z_size,))
      if 'goal:proprio' in obs_space.spaces:
        obs_space.spaces['goal:zstate'] = gym.spaces.Box(-1, 1, (preproc.z_size,))
  else:
    env = AsyncVectorEnv([env_fn(G) for _ in range(G.num_envs)])
    tvenv = real_tvenv
    if G.preproc:
      sd = th.load(G.weightdir / f'{G.model}.pt')
      mG = sd.pop('G')
      mG.device = G.device
      preproc = net_map[G.model](tenv, mG)
      preproc.to(G.device)
      preproc.load(G.weightdir)
      for p in preproc.parameters():
        p.requires_grad = False
      preproc.eval()
      env = wrappers.PreprocVecEnv(preproc, env, G)
      real_tvenv = tvenv = wrappers.PreprocVecEnv(preproc, tvenv, G)
      obs_space.spaces['zstate'] = gym.spaces.Box(-1, 1, (preproc.z_size,))
      if 'goal:proprio' in obs_space.spaces:
        obs_space.spaces['goal:zstate'] = gym.spaces.Box(-1, 1, (preproc.z_size,))
  # tenv.reset()
  epoch = -1
  if tenv.__class__.__name__ == 'BodyGoalEnv':
    goal_key = 'goal:proprio'
  elif tenv.__class__.__name__ == 'CubeGoalEnv':
    goal_key = 'goal:object'

  # Create actor-critic module and target networks
  ac = ActorCritic(obs_space, act_space, goal_key, G=G).to(G.device)

  # Experience buffer

  buf = PPOBuffer(G, obs_space=obs_space, act_space=act_space, size=G.num_envs * G.steps_per_epoch)

  # Count variables (protip: try to get a feel for how different size networks behave!)
  var_counts = tuple(utils.count_vars(module) for module in [ac.pi, ac.v])
  print('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)
  sum_count = sum(var_counts)

  def compute_loss_pi(data):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
    pi, logp = ac.pi(obs, act)
    ratio = th.exp(logp - logp_old)
    clip_adv = th.clamp(ratio, 1 - G.clip_ratio, 1 + G.clip_ratio) * adv
    loss_pi = -(th.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().cpu()
    ent = pi.entropy().mean().cpu()
    clipped = ratio.gt(1 + G.clip_ratio) | ratio.lt(1 - G.clip_ratio)
    clipfrac = th.as_tensor(clipped, dtype=th.float32).mean().cpu()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

  # Set up function for computing value loss
  def compute_loss_v(data):
    obs, ret = data['obs'], data['ret']
    return ((ac.v(obs) - ret)**2).mean()

  # Set up optimizers for policy and value function
  pi_optimizer = Adam(ac.pi.parameters(), lr=G.pi_lr, betas=(0.9, 0.999), eps=1e-8)
  vf_optimizer = Adam(ac.v.parameters(), lr=G.vf_lr, betas=(0.9, 0.999), eps=1e-8)

  def update(data):

    pi_l_old, pi_info_old = compute_loss_pi(data)
    pi_l_old = pi_l_old.cpu()
    v_l_old = compute_loss_v(data).cpu()

    # Train policy with multiple steps of gradient descent
    for i in range(G.train_pi_iters):
      idxs = np.random.randint(0, data['act'].shape[0], G.bs)
      pi_optimizer.zero_grad()
      loss_pi, pi_info = compute_loss_pi(tree_map(lambda x: x[idxs], data))
      kl = pi_info['kl']
      #if kl > 1.5 * G.target_kl:
      #  break
      loss_pi.backward()
      pi_optimizer.step()

    logger['StopIter'] += [i]

    # Value function learning
    for i in range(G.train_v_iters):
      idxs = np.random.randint(0, data['act'].shape[0], G.bs)
      vf_optimizer.zero_grad()
      loss_v = compute_loss_v(tree_map(lambda x: x[idxs], data))
      loss_v.backward()
      vf_optimizer.step()

    # Log changes from updte
    kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    logger['LossPi'] += [pi_l_old.detach().cpu()]
    logger['LossV'] += [v_l_old.detach().cpu()]
    logger['KL'] += [kl.detach().cpu()]
    logger['Entropy'] += [ent.detach().cpu()]
    logger['ClipFrac'] += [cf.detach().cpu()]
    logger['DeltaLossPi'] += [loss_pi.detach().cpu() - pi_l_old.detach().cpu()]
    logger['DeltaLossV'] += [loss_v.detach().cpu() - v_l_old.detach().cpu()]

  def test_agent(itr, use_lenv=False):
    # init
    REP = 4
    if use_lenv:
      pf = th
      _env = learned_tvenv
      o, ep_ret, ep_len = _env.reset(), th.zeros(TN).to(G.device), th.zeros(TN).to(G.device)
    else:
      pf = np
      _env = real_tvenv
      o, ep_ret, ep_len = _env.reset(), np.zeros(TN), np.zeros(TN)

    # run
    frames = []
    dones = []
    rs = []
    vs = []
    all_done = pf.zeros_like(ep_ret)
    success = pf.zeros_like(ep_ret)
    for i in range(G.ep_len):
      # Take deterministic actions at test time
      a, v, logp = ac.step(o)
      if not use_lenv and G.lenv:
        a = a.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
      o, r, d, info = _env.step(a)
      all_done = pf.logical_or(all_done, d)
      if i != (G.ep_len - 1):
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
      frames = frames.transpose(0, 2, 1, 3, 4).reshape([-1, G.lcd_h * 1 * REP, TN * G.lcd_w * REP, 3])
      # make borders
      for k in range(TN):
        if use_lenv:
          frames[:, :, k * REP * G.lcd_w] = [0, 0, 1]
        else:
          frames[:, :, k * REP * G.lcd_w] = [1, 0, 0]
        #frames[:, :, k * REP * G.lcd_w] = 0.0

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
            color = yellow if dones[i][j].detach().cpu().numpy() and i != G.ep_len - 1 else white
            draw.text((G.lcd_w * REP * j + 10, 10), f't: {i} r:{rs[i][j].detach().cpu().numpy():.3f}\nV: {vs[i][j].detach().cpu().numpy():.3f}', fill=color, fnt=fnt)
            draw.text((G.lcd_w * REP * j + 5, 5), f'{"*"*int(success[j].detach().cpu().numpy())}', fill=yellow, fnt=fnt)
            #draw.text((G.lcd_w * REP * (j+1) - 20, 10), '[]', fill=purple, fnt=fnt)
          else:
            color = yellow if dones[i][j] and i != G.ep_len - 1 else white
            draw.text((G.lcd_w * REP * j + 10, 10), f't: {i} r:{rs[i][j]:.3f}\nV: {vs[i][j]:.3f}', fill=color, fnt=fnt)
            draw.text((G.lcd_w * REP * j + 5, 5), f'{"*"*int(success[j])}', fill=yellow, fnt=fnt)
        dframes += [np.array(pframe)]
      dframes = np.stack(dframes)
      vid = dframes.transpose(0, -1, 1, 2)[None]
      utils.add_video(writer, f'{prefix}_rollout', vid, itr + 1, fps=G.fps)
      print('wrote video', prefix)
    logger[f'{prefix}_test/EpRet'] += [proc(ep_ret).mean()]
    logger[f'{prefix}_test/EpLen'] += [proc(ep_len).mean()]
    logger[f'{prefix}_test/success_rate'] += [proc(success).mean()]
  test_agent(-1)
  if G.lenv:
    test_agent(-1, use_lenv=True)
  # writer.flush()

  # Prepare for interaction with environment
  epoch_time = start_time = time.time()

  if G.lenv and G.lenv_cont_roll:
    import ipdb; ipdb.set_trace()
    o = env.reset(np.arange(G.num_envs))
    for itr in itertools.count(1):
      o = {key: val for key, val in o.items()}
      a, v, logp = ac.step(o)
      # Step the venv
      next_o, r, d, info = env.step(a)
      a = get_action(o).detach()
      o2, rew, done, info = env.step(a)
      batch = {'obs': o, 'act': a, 'rew': rew, 'obs2': o2, 'done': done}
      batch = tree_map(lambda x: x.detach(), batch)
      update(batch)
      o = o2

      if itr % G.steps_per_epoch == 0:
        update(data)
        epoch = itr // G.log_n
        if epoch % G.test_n == 0:
          test_agent(itr)
          if G.lenv:
            test_agent(itr, use_lenv=True)
        if epoch % G.save_n == 0:
          ac.save(G.logdir)
        # Log info about epoch
        print('=' * 30)
        print('Epoch', epoch)
        logger['var_count'] = [sum_count]
        logger['dt'] = dt = time.time() - epoch_time
        for key in logger:
          val = np.mean(logger[key])
          writer.add_scalar(key, val, itr)
          print(key, val)
        writer.flush()
        print('Time', time.time() - start_time)
        print('dt', dt)
        print(G.logdir)
        print(G.full_cmd)
        print('=' * 30)
        logger = defaultdict(lambda: [])
        epoch_time = time.time()
        with open(G.logdir / 'hps.yaml', 'w') as f:
          yaml.dump(G, f, width=1000)


  if G.lenv:
    o, ep_ret, ep_len = env.reset(np.arange(G.num_envs)), th.zeros(G.num_envs).to(G.device), np.zeros(G.num_envs)#.to(G.device)
    success = th.zeros(G.num_envs).to(G.device)
    time_to_succ = G.ep_len * th.ones(G.num_envs).to(G.device)
    pf = th
  else:
    o, ep_ret, ep_len = env.reset(), np.zeros(G.num_envs), np.zeros(G.num_envs)
    success = np.zeros(G.num_envs, dtype=np.bool)
    time_to_succ = G.ep_len * np.ones(G.num_envs)
    pf = np
  # Main loop: collect experience in venv and updte/log each epoch
  for itr in range(1, G.total_steps+1):
    with utils.Timer(logger, 'action'):
      o = {key: val for key, val in o.items()}
      a, v, logp = ac.step(o)
    # Step the venv
    with utils.Timer(logger, 'step'):
      next_o, r, d, info = env.step(a)#, logger)
    ep_ret += r
    ep_len += 1

    # store
    trans = {'act': a, 'rew': r, 'val': v, 'logp': logp}
    for key in o:
      trans[f'o:{key}'] = o[key]
    if G.lenv:
      def fx(x):
        if isinstance(x, np.ndarray):
          return x
        else:
          return x.detach().cpu().numpy()
      trans = tree_map(fx, trans)
    buf.store_n(trans)

    o = next_o

    if G.lenv:
      d = d.cpu().numpy()
      def proc(x): return x.cpu().float()
    else:
      def proc(x): return x
    timeout = ep_len == G.ep_len
    terminal = np.logical_or(d, timeout)
    epoch_ended = itr % G.steps_per_epoch == 0
    terminal_epoch = np.logical_or(terminal, epoch_ended)
    timeout_epoch = np.logical_or(timeout, epoch_ended)
    mask = ~timeout_epoch
    if G.learned_rew:
      #logger['preproc_rew'] += [info['preproc_rew'].mean()] 
      logger['learned_rew'] += [info['learned_rew'].mean()] 
      logger['og_rew'] += [info['og_rew'].mean()] 
      logger['goal_delta'] += [info['goal_delta'].mean()] 
      logger['rew_delta'] += [info['rew_delta'].mean()] 
    # if trajectory didn't reach terminal state, bootstrap value target
    _, v, _ = ac.step(o)
    v[mask] *= 0
    buf.finish_paths(np.nonzero(terminal_epoch)[0], v)
    for idx in np.nonzero(terminal_epoch)[0]:
      logger['EpRet'] += [proc(ep_ret[idx])]
      logger['EpLen'] += [ep_len[idx]]
      ep_ret[idx] = 0
      ep_len[idx] = 0

    if epoch_ended:
      if (G.logdir / 'pause.marker').exists():
        import ipdb; ipdb.set_trace()
      epoch = itr // G.steps_per_epoch
      update(buf.get())
      with utils.Timer(logger, 'test_agent'):
        test_agent(itr)
        if G.lenv:
          test_agent(itr, use_lenv=True)

      # save it
      ac.save(G.logdir)

      # Log info about epoch
      print('=' * 30)
      print('Epoch', epoch)
      logger['var_count'] = [sum_count]
      logger['dt'] = dt = time.time() - epoch_time
      logger['env_interactions'] = env_interactions = itr * G.num_envs
      for key in logger:
        print(key, end=' ')
        val = np.mean(logger[key])
        writer.add_scalar(key, val, itr + 1)
        print(val)
      writer.flush()
      print('TotalEnvInteracts', env_interactions)
      print('Time', time.time() - start_time)
      print('dt', dt)
      print(G.logdir)
      print(G.full_cmd)
      print('=' * 30)
      logger = defaultdict(lambda: [])
      epoch_time = time.time()
      with open(G.logdir / 'hps.yaml', 'w') as f:
        yaml.dump(G, f, width=1000)


_G = boxLCD.utils.AttrDict()
_G.replay_size = int(1e6)
_G.total_steps = 1000000000
_G.test_n = 1
_G.save_freq = 10
_G.gamma = 0.99
_G.learned_alpha = 1
_G.pi_lr = 3e-4
_G.vf_lr = 1e-3
_G.alpha = 0.1  # for ppo w/o learned alpha
_G.polyak = 0.995
_G.num_test_episodes = 2
_G.update_every = 40
_G.start_steps = 1000
_G.update_after = 1000
_G.use_done = 0
_G.net = 'mlp'
_G.zdelta = 1
_G.lenv = 0
_G.lenv_mode = 'swap'
_G.lenv_temp = 1.0
_G.lenv_cont_roll = 0
_G.lenv_goals = 0
_G.reset_prompt = 1
_G.succ_reset = 1  # between lenv and normal env
_G.state_key = 'proprio'
_G.diff_delt = 0
_G.goal_thresh = 0.05
_G.preproc_rew = 0
_G.learned_rew = 0
_G.clip_ratio = 0.2
_G.train_pi_iters = 80
_G.train_v_iters = 80
_G.lam = 0.97
_G.steps_per_epoch = 4000
_G.target_kl = 0.01

# TODO: allow changing default values.


if __name__ == '__main__':
  print('TODO: metric to compare env vs. algo runtime. where is bottleneck?')
  import argparse
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  for key, value in _G.items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  tempC = parser.parse_args()
  # grab defaults from the env
  if tempC.env in env_map:
    Env = env_map[tempC.env]
    parser.set_defaults(**Env.ENV_DG)
    parser.set_defaults(**{'goals': 1, 'autoreset': 1})
  G = parser.parse_args()
  G.lcd_w = int(G.wh_ratio * G.lcd_base)
  G.lcd_h = G.lcd_base
  G.imsize = G.lcd_w * G.lcd_h
  # RUN
  ppo(G)
