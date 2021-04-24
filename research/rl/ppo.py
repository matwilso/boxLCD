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
from buffers import OGRB, ReplayBuffer, PPOBuffer
from pponets import ActorCritic
from research.define_config import config, args_type, env_fn
from boxLCD import env_map
import boxLCD
from research import utils
from research import wrappers
#from research.nets.flat_everything import FlatEverything
from jax.tree_util import tree_multimap, tree_map
from research.nets import net_map

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
  real_tvenv = wrappers.AsyncVectorEnv([env_fn(G) for _ in range(TN)])
  if G.lenv:
    sd = th.load(G.weightdir / f'{G.model}.pt')
    mG = sd.pop('G')
    mG.device = G.device
    model = net_map[G.model](tenv, mG)
    model.to(G.device)
    model.eval()
    env = wrappers.RewardLenv(wrappers.LearnedEnv(G.num_envs, model, G))
    tvenv = learned_tvenv = wrappers.RewardLenv(wrappers.LearnedEnv(TN, model, G))
    obs_space.spaces = utils.subdict(obs_space.spaces, env.observation_space.spaces.keys())
  else:
    env = wrappers.AsyncVectorEnv([env_fn(G) for _ in range(G.num_envs)])
    tvenv = real_tvenv
    if G.preproc:
      MC = th.load(G.weightdir / 'bvae.pt').pop('G')
      preproc = BVAE(tenv, MC)
      preproc.load(G.weightdir)
      for p in preproc.parameters():
        p.requires_grad = False
      preproc.eval()

      env = wrappers.PreprocVecEnv(preproc, env, G)
      real_tvenv = tvenv = wrappers.PreprocVecEnv(preproc, tvenv, G)
      obs_space.spaces['zstate'] = gym.spaces.Box(-1, 1, (preproc.z_size,))
      # if 'goal:proprio' in obs_space.spaces:
      #  obs_space.spaces['goal:zstate'] = gym.spaces.Box(-1, 1, (preproc.z_size,))

  # tenv.reset()
  epoch = -1

  # Create actor-critic module and target networks
  ac = ActorCritic(obs_space, act_space, G=G).to(G.device)
  ac_targ = deepcopy(ac)

  # Freeze target networks with respect to optimizers (only updte via polyak averaging)
  for p in ac_targ.parameters():
    p.requires_grad = False

  # List of parameters for both Q-networks (save this for convenience)
  q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

  # Experience buffer
  buf = PPOBuffer(G, obs_space=obs_space, act_space=act_space)

  # Count variables (protip: try to get a feel for how different size networks behave!)
  var_counts = tuple(utils.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
  print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)
  sum_count = sum(var_counts)

  def compute_loss_pi(data):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
    pi, logp = ac.pi(obs, act)
    ratio = th.exp(logp - logp_old)
    clip_adv = th.clamp(ratio, 1 - G.clip_ratio, 1 + G.clip_ratio) * adv
    loss_pi = -(th.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1 + G.clip_ratio) | ratio.lt(1 - G.clip_ratio)
    clipfrac = th.as_tensor(clipped, dtype=th.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

  # Set up function for computing value loss
  def compute_loss_v(data):
    obs, ret = data['obs'], data['ret']
    return ((ac.v(obs) - ret)**2).mean()

  # Set up optimizers for policy and value function
  pi_optimizer = Adam(ac.pi.parameters(), lr=G.pi_lr, betas=(0.9, 0.999), eps=1e-8)
  vf_optimizer = Adam(ac.v.parameters(), lr=G.vf_lr, betas=(0.9, 0.999), eps=1e-8)
  if G.learned_alpha:
    alpha_optimizer = Adam([ac.log_alpha], lr=G.alpha_lr)

  def update():
    data = buf.get()

    pi_l_old, pi_info_old = compute_loss_pi(data)
    pi_l_old = pi_l_old.item()
    v_l_old = compute_loss_v(data).item()

    # Train policy with multiple steps of gradient descent
    for i in range(G.train_pi_iters):
      idxs = np.random.randint(0, data['obs'].shape[0], G.bs)
      pi_optimizer.zero_grad()
      loss_pi, pi_info = compute_loss_pi(nest.map_structure(lambda x: x[idxs], data))
      kl = pi_info['kl']
      if kl > 1.5 * G.target_kl:
        break
      loss_pi.backward()
      pi_optimizer.step()

    logger['StopIter'] += [i]

    # Value function learning
    for i in range(G.train_v_iters):
      idxs = np.random.randint(0, data['obs'].shape[0], G)
      vf_optimizer.zero_grad()
      loss_v = compute_loss_v(nest.map_structure(lambda x: x[idxs], data))
      loss_v.backward()
      vf_optimizer.step()

    # Log changes from update
    kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    logger['LossPi'] += [pi_l_old]
    logger['LossV'] += [v_l_old]
    logger['KL'] += [kl]
    logger['Entropy'] += [ent]
    logger['ClipFrac'] += [cf]
    logger['DeltaLossPi'] += [loss_pi.item() - pi_l_old]
    logger['DeltaLossV'] += [loss_v.item() - v_l_old]

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
      logger[key] += [q_info[key].detach().cpu()]

    # Freeze Q-networks so you don't waste computational effort
    # computing gradients for them during the policy learning step.
    for p in q_params:
      p.requires_grad = False
      if 'vae' in G.net:
        for p in ac.preproc.parameters():
          p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi, loss_alpha, pi_info = compute_loss_pi(data)
    loss_pi.backward()
    logger['Pi/grad_norm'] += [utils.compute_grad_norm(ac.pi.parameters()).detach().cpu()]
    pi_optimizer.step()
    # and optionally the alpha
    if G.learned_alpha:
      alpha_optimizer.zero_grad()
      loss_alpha.backward()
      alpha_optimizer.step()
      logger['LossAlpha'] += [loss_alpha.detach().cpu()]
      logger['Alpha'] += [th.exp(ac.log_alpha.detach().cpu())]

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
      p.requires_grad = True

    # Record things
    logger['LossPi'] += [loss_pi.detach().cpu()]
    for key in pi_info:
      logger[key] += [pi_info[key]]

    # Finally, update target networks by polyak averaging.
    with th.no_grad():
      for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
        # NB: We use an in-place operations "mul_", "add_" to update target
        # params, as opposed to "mul" and "add", which would make new tensors.
        p_targ.data.mul_(G.polyak)
        p_targ.data.add_((1 - G.polyak) * p.data)

  def get_action(o, deterministic=False):
    o = {key: th.as_tensor(1.0 * val, dtype=th.float32).to(G.device) for key, val in o.items()}
    act = ac.act(o, deterministic)
    if not G.lenv:
      act = act.cpu().numpy()
    return act

  def get_action_val(o, deterministic=False):
    with th.no_grad():
      o = {key: th.as_tensor(1.0 * val, dtype=th.float32).to(G.device) for key, val in o.items()}
      a, _, ainfo = ac.pi(o, deterministic, False)
      q = ac.value(o, a)
    if not G.lenv:
      a = a.cpu().numpy()
    return a, q

  def test_agent(itr, use_lenv=False):
    # init
    REP = 4
    if use_lenv:
      pf = th
      _env = learned_tvenv
      o, ep_ret, ep_len = _env.reset(np.arange(TN)), th.zeros(TN).to(G.device), th.zeros(TN).to(G.device)
    else:
      pf = np
      _env = real_tvenv
      o, ep_ret, ep_len = _env.reset(np.arange(TN)), np.zeros(TN), np.zeros(TN)

    # run
    frames = []
    dones = []
    rs = []
    qs = []
    all_done = pf.zeros_like(ep_ret)
    success = pf.zeros_like(ep_ret)
    for i in range(G.ep_len):
      # Take deterministic actions at test time
      a, q = get_action_val(o)
      if not use_lenv and G.lenv:
        a = a.cpu().numpy()
        q = q.cpu().numpy()
      o, r, d, info = _env.step(a)
      all_done = pf.logical_or(all_done, d)
      if i != (G.ep_len - 1):
        success = pf.logical_or(success, d)
      rs += [r]
      qs += [q]
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
        frames = frames.cpu().numpy()
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
            color = yellow if dones[i][j].cpu().numpy() and i != G.ep_len - 1 else white
            draw.text((G.lcd_w * REP * j + 10, 10), f't: {i} r:{rs[i][j].cpu().numpy():.3f}\nQ: {qs[i][j].cpu().numpy():.3f}', fill=color, fnt=fnt)
            draw.text((G.lcd_w * REP * j + 10, 10), f'{"*"*int(success[j].cpu().numpy())}', fill=yellow, fnt=fnt)
            #draw.text((G.lcd_w * REP * (j+1) - 20, 10), '[]', fill=purple, fnt=fnt)
          else:
            color = yellow if dones[i][j] and i != G.ep_len - 1 else white
            draw.text((G.lcd_w * REP * j + 10, 10), f't: {i} r:{rs[i][j]:.3f}\nQ: {qs[i][j]:.3f}', fill=color, fnt=fnt)
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
    # TODO: try this with backpropping through stuff
    o = env.reset(np.arange(G.num_envs))
    for itr in itertools.count(1):
      a = get_action(o).detach()
      o2, rew, done, info = env.step(a)
      batch = {'obs': o, 'act': a, 'rew': rew, 'obs2': o2, 'done': done}
      batch = tree_map(lambda x: x.detach(), batch)
      update(batch)
      o = o2
      #success = info['success']
      # env._reset_goals(success)
      # print(itr)
      if itr % 200 == 0:
        o = env.reset()
      if itr % G.log_n == 0:
        epoch = itr // G.log_n
        if epoch % G.test_n == 0:
          test_agent(itr)
          if G.lenv:
            test_agent(itr, use_lenv=True)
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
    o, ep_ret, ep_len = env.reset(np.arange(G.num_envs)), th.zeros(G.num_envs).to(G.device), th.zeros(G.num_envs).to(G.device)
    success = th.zeros(G.num_envs).to(G.device)
    time_to_succ = G.ep_len * th.ones(G.num_envs).to(G.device)
    pf = th
  else:
    o, ep_ret, ep_len = env.reset(np.arange(G.num_envs)), np.zeros(G.num_envs), np.zeros(G.num_envs)
    success = np.zeros(G.num_envs, dtype=np.bool)
    time_to_succ = G.ep_len * np.ones(G.num_envs)
    pf = np
  # Main loop: collect experience in venv and update/log each epoch
  for itr in range(G.total_steps):
    # Until start_steps have elapsed, randomly sample actions
    # from a uniform distribution for better exploration. Afterwards,
    # use the learned policy.
    if itr > G.start_steps:
      with utils.Timer(logger, 'action'):
        o = {key: val for key, val in o.items()}
        a = get_action(o)
    else:
      a = env.action_space.sample()

    # Step the venv
    o2, r, d, info = env.step(a)
    ep_ret += r
    ep_len += 1

    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    d[ep_len == G.ep_len] = False
    success = pf.logical_or(success, d)
    time_to_succ = pf.minimum(time_to_succ, G.ep_len * ~success + ep_len * success)
    # print(time_to_succ)

    # Store experience to replay buffer
    trans = {'act': a, 'rew': r, 'done': d}
    for key in o:
      trans[f'o:{key}'] = o[key]
    for key in o2:
      trans[f'o2:{key}'] = o2[key]
    buf.store_n(trans)

    # Super critical, easy to overlook step: make sure to updte
    # most recent observation!
    o = o2

    # End of trajectory handling
    if G.lenv:
      done = th.logical_or(d, ep_len == G.ep_len)
      dixs = th.nonzero(done)
      def proc(x): return x.cpu().float()
    else:
      done = np.logical_or(d, ep_len == G.ep_len)
      dixs = np.nonzero(done)[0]
      def proc(x): return x
    if len(dixs) == G.num_envs or (not G.lenv and G.succ_reset):
      # if not G.lenv or len(dixs) == G.num_envs:
      for idx in dixs:
        logger['EpRet'] += [proc(ep_ret[idx])]
        logger['EpLen'] += [proc(ep_len[idx])]
        logger['success_rate'] += [proc(success[idx])]
        logger['time_to_succ'] += [proc(time_to_succ[idx])]
        ep_ret[idx] = 0
        ep_len[idx] = 0
        success[idx] = 0
        time_to_succ[idx] = G.ep_len
        #logger['success_rate'] += [proc(d[idx])]
        #logger['success_rate_nenv'] += [proc(d[idx])**(1/G.num_envs)]
      if len(dixs) != 0:
        o = env.reset(dixs)
        if G.lenv:
          assert len(dixs) == G.num_envs, "the learned env needs the envs to be in sync in terms of history.  you can't easily reset one without resetting the others. it's hard"
        else:
          assert env.shared_memory, "i am not sure if this works when you don't do shared memory. it would need to be tested. something like the comment below"
        #o = tree_multimap(lambda x,y: ~done[:,None]*x + done[:,None]*y, o, reset_o)

    # Updte handling
    if itr >= G.update_after and itr % G.update_every == 0:
      for j in range(int(G.update_every * 1.0)):
        with utils.Timer(logger, 'sample_batch'):
          batch = buf.sample_batch(G.bs)
        with utils.Timer(logger, 'updte'):
          update(data=batch)

    # End of epoch handling
    if (itr + 1) % G.log_n == 0:
      epoch = (itr + 1) // G.log_n

      # Save model
      if (epoch % G.save_freq == 0) or (itr == G.total_steps):
        th.save(ac, G.logdir / 'weights.pt')

      if (G.logdir / 'pause.marker').exists():
        import ipdb; ipdb.set_trace()

      # Test the performance of the deterministic version of the agent.
      if epoch % G.test_n == 0:
        with utils.Timer(logger, 'test_agent'):
          test_agent(itr)
          if G.lenv:
            test_agent(itr, use_lenv=True)

        if 'vae' in G.net:
          test_batch = buf.sample_batch(8)
          ac.preproc.evaluate(writer, test_batch['obs'], itr)
        #test_agent(video=epoch % 1 == 0)
        # if buf.ptr > G.ep_len*4:
        #  eps = buf.get_last(4)
        #  goal = eps['obs']['goal:lcd']
        #  lcd = eps['obs']['lcd']
        #  goal = goal.reshape([4, -1, *goal.shape[1:]])
        #  lcd = lcd.reshape([4, -1, *lcd.shape[1:]])
        #  error = (goal - lcd + 1) / 2
        #  out = np.concatenate([goal, lcd, error], 2)
        #  out = utils.combine_imgs(out[:,:,None], row=1, col=4)[None,:,None]
        #  utils.add_video(writer, 'samples', out, epoch, fps=G.fps)

      # Log info about epoch
      print('=' * 30)
      print('Epoch', epoch)
      logger['var_count'] = [sum_count]
      logger['dt'] = dt = time.time() - epoch_time
      logger['env_interactions'] = env_interactions = itr * G.num_envs
      for key in logger:
        val = np.mean(logger[key])
        writer.add_scalar(key, val, itr + 1)
        print(key, val)
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


_C = boxLCD.utils.AttrDict()
_C.replay_size = int(1e6)
_C.total_steps = 1000000
_C.test_n = 1
_C.save_freq = 10
_C.gamma = 0.99
_C.learned_alpha = 1
_C.alpha_lr = 1e-4  # for ppo w/ learned alpha
_C.alpha = 0.1  # for ppo w/o learned alpha
_C.polyak = 0.995
_C.num_test_episodes = 2
_C.update_every = 40
_C.start_steps = 1000
_C.update_after = 1000
_C.use_done = 0
_C.net = 'mlp'
_C.zdelta = 1
_C.lenv = 0
_C.lenv_mode = 'swap'
_C.lenv_temp = 1.0
_C.lenv_cont_roll = 0
_C.lenv_goals = 0
_C.reset_prompt = 0
_C.succ_reset = 1  # between lenv and normal env
_C.state_key = 'proprio'
_C.diff_delt = 0
_C.goal_thresh = 0.010
_C.preproc_rew = 0
_C.clip_ratio = 0.2
_C.train_pi_iters = 80
_C.train_v_iters = 80
# TODO: allow changing default values.


if __name__ == '__main__':
  print('TODO: metric to compare env vs. algo runtime. where is bottleneck?')
  import argparse
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  for key, value in _C.items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  tempC = parser.parse_args()
  # grab defaults from the env
  if tempC.env in env_map:
    Env = env_map[tempC.env]
    parser.set_defaults(**Env.ENV_DG)
    parser.set_defaults(**{'goals': 1})
  G = parser.parse_args()
  G.lcd_w = int(G.wh_ratio * G.lcd_base)
  G.lcd_h = G.lcd_base
  G.imsize = G.lcd_w * G.lcd_h
  # RUN
  ppo(G)
