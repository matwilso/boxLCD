from research import utils
from research.define_config import config, args_type, env_fn
from boxLCD import env_map
from research.rl.ppo import PPO
from research.rl.sac import SAC

_G = utils.AttrDict()
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
_G.use_done = 1
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
# SAC
_G.replay_size = int(1e6)
_G.total_steps = 1000000
_G.learned_alpha = 1
_G.alpha_lr = 1e-4  # for SAC w/ learned alpha
_G.alpha = 0.1  # for SAC w/o learned alpha
_G.polyak = 0.995

if __name__ == '__main__':
  #print('TODO: metric to compare env vs. algo runtime. where is bottleneck?')
  import argparse
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  for key, value in _G.items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  parser.add_argument('algo')
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
  if G.algo == 'ppo':
    ppo = PPO(G)
    ppo.run()
  elif G.algo == 'sac':
    sac = SAC(G)
    sac.run()