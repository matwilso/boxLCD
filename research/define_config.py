import torch as th
import gym
import argparse
from research import wrappers
import subprocess
import sys
import pathlib
import boxLCD.utils
from boxLCD import envs, env_map
from boxLCD import ENV_DC
from boxLCD.utils import args_type


def env_fn(C, seed=None):
  def _make():
    if C.env in env_map:
      env = env_map[C.env](C)
      env.seed(seed)
      if C.goals:
        if 'Cube' not in C.env:
          env = wrappers.BodyGoalEnv(env, C)
        else:
          env = wrappers.CubeGoalEnv(env, C)
      #if C.preproc:
      #  assert C.env == 'Luxo'
      #  env = wrappers.PreprocEnv(env, C)
    else:
      env = gym.make(C.env)
      env = wrappers.WrappedGym(env, C)
      env.seed(seed)
    return env
  return _make

def config():
  C = boxLCD.utils.AttrDict()
  # BASICS
  C.logdir = pathlib.Path('./logs/trash')
  C.weightdir = pathlib.Path('.')
  C.buffdir = pathlib.Path('.')
  C.datapath = pathlib.Path('.')
  C.device = 'cuda' # 'cuda', 'cpu'
  C.mode = 'train'
  C.model = 'frame_token'
  C.datamode = 'video'
  C.ipython_mode = 0

  #C.data_mode = 'image'
  C.amp = 0
  C.total_itr = int(1e9)
  C.log_n = int(1e4)
  C.save_n = 5
  C.refresh_data = 0

  C.decode = 'multi'
  C.conv_io = 0
  C.train_barrels = -1  # -1 means all. any other number is how many to use
  C.test_barrels = 1 
  C.grad_clip = 10.0

  C.bs = 64
  C.lr = 1e-4
  C.n_layer = 2
  C.n_head = 4
  C.n_embed = 128
  C.hidden_size = 128
  C.nfilter = 128
  C.vidstack = -1
  C.stacks_per_block = 32

  C.vqD = 128
  C.vqK = 128
  C.beta = 0.25


  C.min_std = 1e-4
  C.data_frac = 1.0
  C.vanished = 1
  C.num_envs = 8

  C.mdn_k = 5
  C.dist_delta = 0
  C.sample_sample = 0
  C.skip_train = 0

  C.phase = 1
  C.window = 200
  C.seed = 0
  C.end2end = 0

  C.env = 'Dropbox'
  C.goals = 0
  C.preproc = 0
  C.state_rew = 1
  C.rew_scale = 1.0

  # extra info that we set here for convenience and don't modify 
  C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
  C.commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')

  # values set by the code
  C.num_vars = 0

  pastKeys = list(C.keys())
  for key, val in ENV_DC.items():
    assert key not in pastKeys, f'make sure you are not duplicating keys {key}'
    C[key] = val

  


  return C