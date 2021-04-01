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
from buffers import OGRB, ReplayBuffer
from sacnets import ActorCritic
from research.define_config import config, args_type, env_fn
from boxLCD import env_map
import boxLCD
from research import utils
from async_vector_env import AsyncVectorEnv
from research.wrappers import RewardGoalEnv
from research.learned_env import LearnedEnv, RewardLenv
from research.nets.flat_everything import FlatEverything

def sac(C):
  print(C.full_cmd)
  # th.manual_seed(C.seed)
  # np.random.seed(C.seed)

  # Set up logger and save configuration
  logger = defaultdict(lambda: [])
  writer = SummaryWriter(C.logdir)
  tenv = env_fn(C, C.seed)()  # test env
  obs_space = tenv.observation_space
  act_space = tenv.action_space
  TN = 4
  real_tvenv = AsyncVectorEnv([env_fn(C) for _ in range(TN)])
  if C.lenv:
    MC = th.load(C.weightdir / 'flatev2.pt').pop('C')
    model = FlatEverything(tenv, MC)
    env = RewardLenv(LearnedEnv(C.num_envs, model, C))
    tvenv = learned_tvenv = RewardLenv(LearnedEnv(TN, model, C))
    obs_space.spaces = utils.subdict(obs_space.spaces, env.observation_space.spaces.keys())
  else:
    env = AsyncVectorEnv([env_fn(C) for _ in range(C.num_envs)])
    tvenv = real_tvenv

  #tenv.reset()
  epoch = -1

  # Create actor-critic module and target networks
  ac = ActorCritic(tenv.observation_space, tenv.action_space, C=C).to(C.device)
  ac_targ = deepcopy(ac)

  # Freeze target networks with respect to optimizers (only update via polyak averaging)
  for p in ac_targ.parameters():
    p.requires_grad = False

  # List of parameters for both Q-networks (save this for convenience)
  q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

  # Experience buffer
  replay_buffer = ReplayBuffer(C, obs_space=obs_space, act_space=act_space)

  # Count variables (protip: try to get a feel for how different size networks behave!)
  var_counts = tuple(utils.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
  print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)
  sum_count = sum(var_counts)

  # Set up function for computing SAC Q-losses
  def compute_loss_q(data):
    alpha = C.alpha if not C.learned_alpha else th.exp(ac.log_alpha).detach()
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
    if not C.use_done:
      d = 0
    if C.net == 'vae' and C.vae_rew:
      r = ac.comp_rew(o).detach()
    q1 = ac.q1(o, a)
    q2 = ac.q2(o, a)

    # Bellman backup for Q functions
    with th.no_grad():
      # Target actions come from *current* policy
      a2, logp_a2, ainfo = ac.pi(o2)

      # Target Q-values
      q1_pi_targ = ac_targ.q1(o2, a2)
      q2_pi_targ = ac_targ.q2(o2, a2)
      q_pi_targ = th.min(q1_pi_targ, q2_pi_targ)
      backup = r + C.gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2

    # Useful info for logging
    q_info = dict(Q1Vals=q1.mean().detach().cpu(), Q2Vals=q2.mean().detach().cpu())
    q_info['batchR'] = r.mean().detach().cpu()
    return loss_q, q_info

  # Set up function for computing SAC pi loss
  def compute_loss_pi(data):
    alpha = C.alpha if not C.learned_alpha else th.exp(ac.log_alpha).detach()
    o = data['obs']
    pi, logp_pi, ainfo = ac.pi(o)
    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = th.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (alpha * logp_pi - q_pi).mean()

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.mean().detach().cpu(), action_abs=ainfo['mean'].abs().mean().detach().cpu(), action_std=ainfo['std'].mean().detach().cpu())

    if C.learned_alpha:
      loss_alpha = (-1.0 * (th.exp(ac.log_alpha) * (logp_pi + ac.target_entropy).detach())).mean()
      #loss_alpha = -(ac.log_alpha * (logp_pi + ac.target_entropy).detach()).mean()
    else:
      loss_alpha = 0.0

    return loss_pi, loss_alpha, pi_info

  # Set up optimizers for policy and q-function
  q_optimizer = Adam(q_params, lr=C.lr)
  pi_optimizer = Adam(ac.pi.parameters(), lr=C.lr)
  if C.learned_alpha:
    alpha_optimizer = Adam([ac.log_alpha], lr=C.alpha_lr)

  def update(data):
    # TODO: optimize this by not requiring the items right away.
    # i think this might be blockin for pytorch to finish some computations

    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad()
    loss_q, q_info = compute_loss_q(data)
    loss_q.backward()
    logger['Q/grad_norm'] += [utils.compute_grad_norm(ac.q1.parameters()).detach().cpu()]
    q_optimizer.step()

    # Record things
    logger['LossQ'] += [loss_q.detach().cpu()]
    for key in q_info:
      logger[key] += [q_info[key]]

    # Freeze Q-networks so you don't waste computational effort
    # computing gradients for them during the policy learning step.
    for p in q_params:
      p.requires_grad = False
      if C.net == 'vae':
        for p in ac.preproc.parameters():
          p.requires_grad = False


    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi, loss_alpha, pi_info = compute_loss_pi(data)
    loss_pi.backward()
    logger['Pi/grad_norm'] += [utils.compute_grad_norm(ac.pi.parameters()).detach().cpu()]
    pi_optimizer.step()
    # and optionally the alpha
    if C.learned_alpha:
      alpha_optimizer.zero_grad()
      loss_alpha.backward()
      alpha_optimizer.step()
      logger['LossAlpha'] += [loss_alpha.detach().cpu()]
      logger['Alpha'] += [th.exp(ac.log_alpha.detach().cpu())]

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
      p.requires_grad = True

    # Record things
    logger['LossPi'] += [loss_pi.item()]
    for key in pi_info:
      logger[key] += [pi_info[key]]

    # Finally, update target networks by polyak averaging.
    with th.no_grad():
      for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
        # NB: We use an in-place operations "mul_", "add_" to update target
        # params, as opposed to "mul" and "add", which would make new tensors.
        p_targ.data.mul_(C.polyak)
        p_targ.data.add_((1 - C.polyak) * p.data)

  def get_action(o, deterministic=False):
    o = {key: th.as_tensor(1.0*val, dtype=th.float32).to(C.device) for key, val in o.items()}
    act = ac.act(o, deterministic)
    if not C.lenv:
      act = act.cpu().numpy()
    return act

  def get_action_val(o, deterministic=False):
    with th.no_grad():
      o = {key: th.as_tensor(1.0*val, dtype=th.float32).to(C.device) for key, val in o.items()}
      a, _, ainfo = ac.pi(o, deterministic, False)
      q = ac.value(o, a)
    if not C.lenv:
      a = a.cpu().numpy()
    return a, q

  def test_agent(use_lenv=False):
    # init
    REP = 4
    if use_lenv:
      pf = th
      _env = learned_tvenv
      o, ep_ret, ep_len = _env.reset(np.arange(TN)), th.zeros(TN).to(C.device), th.zeros(TN).to(C.device)
    else:
      pf = np
      _env = real_tvenv
      o, ep_ret, ep_len = _env.reset(np.arange(TN)), np.zeros(TN), np.zeros(TN)

    # run
    frames = []
    dones = []
    rs = []
    all_done = pf.zeros_like(ep_ret)
    for i in range(C.ep_len):
      # Take deterministic actions at test time
      a, q = get_action_val(o)
      if not use_lenv:
        a = a.cpu().numpy()
      o, r, d, info = _env.step(a)
      all_done = pf.logical_or(all_done, d)
      rs += [r]
      dones += [d]
      ep_ret += r * ~all_done
      ep_len += 1 * ~all_done
      delta = (1.0 * o['lcd'] - 1.0 * o['goal:lcd'] + 1) / 2
      frame = delta
      #frame = np.concatenate([1.0 * o['goal:lcd'], 1.0 * o['lcd'], delta], axis=-2)
      frames += [frame]

    if use_lenv:
      frames = th.stack(frames)
      frames = frames.cpu().numpy()
    else:
      frames = np.stack(frames)
    frames = frames[..., None].repeat(REP, -3).repeat(REP, -2).repeat(3, -1)
    frames = frames.transpose(0, 2, 1, 3, 4).reshape([-1, C.lcd_h*1*REP, TN*C.lcd_w*REP, 3])
    for k in range(TN):
      frames[:,:,k*REP*C.lcd_w] = 0.0

    dframes = []
    for i in range(len(frames)):
      frame = frames[i]
      pframe = Image.fromarray((frame * 255).astype(np.uint8))
      # get a drawing context
      draw = ImageDraw.Draw(pframe)
      fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 60)
      for j in range(TN):
        if use_lenv:
          color = (255, 255, 50) if dones[i][j].cpu().numpy() and i != C.ep_len-1 else (255, 255, 255)
          draw.text((C.lcd_w*REP*j + 10, 10), f't: {i} r: {rs[i][j].cpu().numpy():.2f}', fill=color, fnt=fnt)
        else:
          color = (255, 255, 50) if dones[i][j] and i != C.ep_len-1 else (255, 255, 255)
          draw.text((C.lcd_w*REP*j + 10, 10), f't: {i} r: {rs[i][j]:.2f}', fill=color, fnt=fnt)
      dframes += [np.array(pframe)]
    dframes = np.stack(dframes)
    vid = dframes.transpose(0, -1, 1, 2)[None]
    utils.add_video(writer, f'rollout', vid, epoch, fps=C.fps)
    if use_lenv:
      proc = lambda x: x.detach().cpu()
    else:
      proc = lambda x: x
    prefix = 'learned' if use_lenv else 'real'
    logger[f'{prefix}_test/EpRet'] += [proc(ep_ret).mean()]
    logger[f'{prefix}_test/EpLen'] += [proc(ep_len).mean()]
    print('wrote video', prefix)
  test_agent()
  if C.lenv: test_agent(use_lenv=True)

  # Prepare for interaction with environment
  total_steps = C.steps_per_epoch * C.epochs
  epoch_time = start_time = time.time()
  if C.lenv:
    o, ep_ret, ep_len = env.reset(np.arange(C.num_envs)), th.zeros(C.num_envs).to(C.device), th.zeros(C.num_envs).to(C.device)
  else:
    o, ep_ret, ep_len = env.reset(np.arange(C.num_envs)), np.zeros(C.num_envs), np.zeros(C.num_envs)
  # Main loop: collect experience in venv and update/log each epoch
  for t in range(total_steps):
    # Until start_steps have elapsed, randomly sample actions
    # from a uniform distribution for better exploration. Afterwards,
    # use the learned policy.
    if t > C.start_steps:
      with utils.Timer(logger, 'action'):
        o = {key: val for key, val in o.items()}
        a = get_action(o)
    else:
      a = env.action_space.sample()

    # Step the venv
    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1

    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    d[ep_len == C.ep_len] = False

    # Store experience to replay buffer
    trans = {'act': a, 'rew': r, 'done': d}
    for key in o:
      trans[f'o:{key}'] = o[key]
    for key in o2:
      trans[f'o2:{key}'] = o2[key]
    replay_buffer.store_n(trans)

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    # End of trajectory handling
    if C.lenv:
      done = th.logical_or(d, ep_len == C.ep_len)
      dixs = th.nonzero(done)
      proc = lambda x: x.cpu()
    else:
      done = np.logical_or(d, ep_len == C.ep_len)
      dixs = np.nonzero(done)[0]
      proc = lambda x: x
    if not C.lenv or len(dixs) == C.num_envs:
      for idx in dixs:
        logger['EpRet'] += [proc(ep_ret[idx])]
        logger['EpLen'] += [proc(ep_len[idx])]
        ep_ret[idx] = 0
        ep_len[idx] = 0
        logger['success_rate'] += [proc(d[idx])]
      if len(dixs) != 0:
        o = env.reset(dixs)
        if C.lenv:
          assert len(dixs) == C.num_envs, "the learned env needs the envs to be in sync in terms of history.  you can't easily reset one without resetting the others. it's hard"
        else:
          assert env.shared_memory, "i am not sure if this works when you don't do shared memory. it would need to be tested. something like the comment below"
        #o = tree_multimap(lambda x,y: ~done[:,None]*x + done[:,None]*y, o, reset_o)

    # Update handling
    if t >= C.update_after and t % C.update_every == 0:
      for j in range(int(C.update_every*1.0)):
        with utils.Timer(logger, 'sample_batch'):
         batch = replay_buffer.sample_batch(C.bs)
        with utils.Timer(logger, 'update'):
          update(data=batch)

    # End of epoch handling
    if (t + 1) % C.steps_per_epoch == 0:
      epoch = (t + 1) // C.steps_per_epoch

      # Save model
      if (epoch % C.save_freq == 0) or (epoch == C.epochs):
        th.save(ac, C.logdir / 'weights.pt')

      if (C.logdir / 'pause.marker').exists():
        import ipdb; ipdb.set_trace()

      # Test the performance of the deterministic version of the agent.
      if epoch % 1 == 0:
        test_agent()
        if C.lenv: test_agent(use_lenv=True)

        if C.net == 'vae':
          test_batch = replay_buffer.sample_batch(8)
          ac.preproc.evaluate(writer, test_batch['obs'], epoch)
        #test_agent(video=epoch % 1 == 0)
        # if replay_buffer.ptr > C.ep_len*4:
        #  eps = replay_buffer.get_last(4)
        #  goal = eps['obs']['goal:lcd']
        #  lcd = eps['obs']['lcd']
        #  goal = goal.reshape([4, -1, *goal.shape[1:]])
        #  lcd = lcd.reshape([4, -1, *lcd.shape[1:]])
        #  error = (goal - lcd + 1) / 2
        #  out = np.concatenate([goal, lcd, error], 2)
        #  out = utils.combine_imgs(out[:,:,None], row=1, col=4)[None,:,None]
        #  utils.add_video(writer, 'samples', out, epoch, fps=C.fps)

      # Log info about epoch
      print('=' * 30)
      print('Epoch', epoch)
      logger['var_count'] = [sum_count]
      logger['dt'] = dt = time.time() - epoch_time
      logger['env_interactions'] = env_interactions = t*C.num_envs
      for key in logger:
        val = np.mean(logger[key])
        writer.add_scalar(key, val, epoch)
        print(key, val)
      writer.flush()
      print('TotalEnvInteracts', env_interactions)
      print('Time', time.time() - start_time)
      print('dt', dt)
      print(C.logdir)
      print(C.full_cmd)
      print('=' * 30)
      logger = defaultdict(lambda: [])
      epoch_time = time.time()
      with open(C.logdir / 'hps.yaml', 'w') as f:
        yaml.dump(C, f)


_C = boxLCD.utils.AttrDict()
_C.replay_size = int(1e6)
_C.epochs = 100000
_C.steps_per_epoch = 4000
_C.save_freq = 10
_C.gamma = 0.99
_C.learned_alpha = 0
_C.alpha_lr = 5e-4  # for SAC w/ learned alpha
_C.alpha = 0.1  # for SAC w/o learned alpha
_C.polyak = 0.995
_C.num_test_episodes = 2
_C.update_every = 40
_C.start_steps = 1000
_C.update_after = 1000
_C.use_done = 0
_C.vae_rew = 0
_C.net = 'mlp'
_C.zdelta = 1
_C.lenv = 0
_C.lenv_mode = 'swap'
_C.lenv_temp = 1.0
_C.reset_prompt = 0 

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  for key, value in _C.items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  tempC = parser.parse_args()
  # grab defaults from the env
  Env = env_map[tempC.env]
  parser.set_defaults(**Env.ENV_DC)
  parser.set_defaults(**{'goals': 1})
  C = parser.parse_args()
  C.lcd_w = int(C.wh_ratio * C.lcd_base)
  C.lcd_h = C.lcd_base
  C.imsize = C.lcd_w * C.lcd_h
  # RUN
  sac(C)
