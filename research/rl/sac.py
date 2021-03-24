import yaml
from datetime import datetime
import PIL
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
from define_config import config, args_type
from boxLCD import env_map
import boxLCD
import utils

def sac(C):
  print(C.full_cmd)
  # th.manual_seed(C.seed)
  # np.random.seed(C.seed)

  def env_fn(C, seed=None):
    def _thunk():
      env = env_map[C.env](C)
      if seed is not None:
        env.seed(seed)
      env = RewardGoalEnv(env)
      return env
    return _thunk

  # Set up logger and save configuration
  logger = defaultdict(lambda: [])
  writer = SummaryWriter(C.logdir)

  env = env_fn(C, C.seed)()
  tenv = env_fn(C, C.seed)()  # test env
  obs_space = tenv.observation_space
  act_space = tenv.action_space
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
  replay_buffer = OGRB(C, obs_space=obs_space, act_space=act_space)

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

    q1 = ac.q1(o, a)
    q2 = ac.q2(o, a)

    # Bellman backup for Q functions
    with th.no_grad():
      # Target actions come from *current* policy
      a2, logp_a2 = ac.pi(o2)

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
    q_info = dict(Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy())

    return loss_q, q_info

  # Set up function for computing SAC pi loss
  def compute_loss_pi(data):
    alpha = C.alpha if not C.learned_alpha else th.exp(ac.log_alpha).detach()
    o = data['obs']
    pi, logp_pi = ac.pi(o)
    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = th.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (alpha * logp_pi - q_pi).mean()

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

    if C.learned_alpha:
      #loss_alpha = (-1.0 * (th.exp(ac.log_alpha) * (logp_pi + ac.target_entropy).detach())).mean()
      loss_alpha = -(ac.log_alpha * (logp_pi + ac.target_entropy).detach()).mean()
    else:
      loss_alpha = 0.0

    return loss_pi, loss_alpha, pi_info

  # Set up optimizers for policy and q-function
  q_optimizer = Adam(q_params, lr=C.vf_lr)
  pi_optimizer = Adam(ac.pi.parameters(), lr=C.pi_lr)
  if C.learned_alpha:
    alpha_optimizer = Adam([ac.log_alpha], lr=C.alpha_lr)

  def update(data):
    # TODO: optimize this by not requiring the items right away.
    # i think this might be blockin for pytorch to finish some computations

    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad()
    loss_q, q_info = compute_loss_q(data)
    loss_q.backward()
    q_optimizer.step()

    # Record things
    logger['LossQ'] += [loss_q.detach().cpu()]
    for key in q_info:
      logger[key] += [q_info[key]]

    # Freeze Q-networks so you don't waste computational effort
    # computing gradients for them during the policy learning step.
    for p in q_params:
      p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi, loss_alpha, pi_info = compute_loss_pi(data)
    loss_pi.backward()
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
    o = {key: th.as_tensor(val, dtype=th.float32).to(C.device) for key, val in o.items()}
    return ac.act(o, deterministic)

  def test_agent(video=False):
    frames = []
    for j in range(C.num_test_episodes):
      o, d, ep_ret, ep_len = tenv.reset(), False, 0, 0
      while not(d or (ep_len == C.ep_len)):
        # Take deterministic actions at test time
        o = {key: val[None] for key, val in o.items()}
        o, r, d, info = tenv.step(get_action(o, False)[0])
        ep_ret += r
        ep_len += 1
        delta = (1.0 * o['lcd'] - 1.0 * o['goal:lcd'] + 1) / 2
        frame = np.c_[1.0 * o['goal:lcd'], 1.0 * o['lcd'], delta]
        frame = frame.repeat(8, 0).repeat(8, 1)[..., None].repeat(3, -1)
        pframe = PIL.Image.fromarray((frame * 255).astype(np.uint8))
        # get a drawing context
        draw = PIL.ImageDraw.Draw(pframe)
        draw.text((10, 10), f'reward: {r:.4f} simi: {info["simi"]:.4f}', fill=(255, 0, 0,))
        frames += [np.array(pframe)]

      logger['TestEpRet'] += [ep_ret]
      logger['TestEpLen'] += [ep_len]
      frames += [np.zeros_like(frames[-1])]
      frames += [np.zeros_like(frames[-1])]
      frames += [np.zeros_like(frames[-1])]

    if len(frames) != 0:
      vid = np.stack(frames)
      vid_tensor = vid.transpose(0, 3, 1, 2)[None]
      utils.add_video(writer, 'rollout', vid_tensor, epoch, fps=C.fps)
      frames = []
      print('wrote video')
  test_agent(video=True)

  # Prepare for interaction with environment
  total_steps = C.steps_per_epoch * C.epochs
  epoch_time = start_time = time.time()
  o, ep_ret, ep_len = env.reset(), 0, 0
  # Main loop: collect experience in venv and update/log each epoch
  for t in range(total_steps):

    # Until start_steps have elapsed, randomly sample actions
    # from a uniform distribution for better exploration. Afterwards,
    # use the learned policy.
    if t > C.start_steps:
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
    d = False if ep_len == C.ep_len else d

    # Store experience to replay buffer
    trans = {'act': a, 'rew': r, 'done': d}
    for key in o:
      trans[f'o:{key}'] = o[key]
    for key in o2:
      trans[f'o2:{key}'] = o2[key]
    replay_buffer.store(trans)

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    # End of trajectory handling
    if d or (ep_len == C.ep_len):
      logger['EpRet'] += [ep_ret]
      logger['EpLen'] += [ep_len]
      o, ep_ret, ep_len = env.reset(), 0, 0

    # Update handling
    if t >= C.update_after and t % C.update_every == 0:
      for j in range(int(C.update_every)):
        batch = replay_buffer.sample_batch(C.bs)
        update(data=batch)

    # End of epoch handling
    if (t + 1) % C.steps_per_epoch == 0:
      epoch = (t + 1) // C.steps_per_epoch

      # Save model
      if (epoch % C.save_freq == 0) or (epoch == C.epochs):
        pass

      # Test the performance of the deterministic version of the agent.
      if epoch % 1 == 0:
        test_agent(video=epoch % 1 == 0)

      # Log info about epoch
      print('=' * 30)
      print('Epoch', epoch)
      logger['var_count'] = [sum_count]
      for key in logger:
        val = np.mean(logger[key])
        writer.add_scalar(key, val, epoch)
        print(key, val)
      writer.flush()
      print('TotalEnvInteracts', t * C.num_envs)
      print('Time', time.time() - start_time)
      print('dt', time.time() - epoch_time)
      print(C.logdir)
      print(C.full_cmd)
      print('=' * 30)
      logger = defaultdict(lambda: [])
      epoch_time = time.time()


_C = boxLCD.utils.AttrDict()
_C.replay_size = int(1e6)
_C.epochs = 100
_C.steps_per_epoch = 4000
_C.save_freq = 10
_C.pi_lr = 1e-3
_C.vf_lr = 1e-3
_C.alpha_lr = 1e-3  # for SAC w/ learned alpha
_C.gamma = 0.99
_C.learned_alpha = 1
_C.alpha = 0.1  # for SAC w/o learned alpha
_C.polyak = 0.995
_C.num_test_episodes = 2
_C.update_every = 50
_C.start_steps = 10000
_C.update_after = 1000
_C.use_done = 1

if __name__ == '__main__':
  import argparse
  from wrappers import RewardGoalEnv
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  for key, value in _C.items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  tempC = parser.parse_args()
  # grab defaults from the env
  Env = env_map[tempC.env]
  parser.set_defaults(**Env.ENV_DC)
  C = parser.parse_args()
  C.lcd_w = int(C.wh_ratio * C.lcd_base)
  C.lcd_h = C.lcd_base
  C.imsize = C.lcd_w * C.lcd_h
  # RUN
  sac(C)
