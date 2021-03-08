import argparse
import subprocess
import sys
import pathlib
from boxLCD import envs, env_map, ENV_DC, wrappers
from boxLCD.utils import args_type, AttrDict

def env_fn(C, seed=None):
  def _make():
    env = env_map[C.env](C)
    # wrap to make lcd show up in observation space
    env = wrappers.LCDEnv(env)
    env.seed(seed)
    return env
  return _make

def config():
  C = AttrDict()
  # BASICS
  C.logdir = pathlib.Path('./logs/')
  C.datapath = pathlib.Path('.')
  C.collect_n = 10000
  C.env = 'Bounce'
  # training stuff
  C.device = 'cuda'  # 'cuda', 'cpu'
  C.num_epochs = 200
  C.bs = 64
  C.lr = 5e-4
  C.n_layer = 2
  C.n_embed = 128
  C.n_head = 4
  # extra info that we set here for convenience and don't modify
  C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
  C.commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
  pastKeys = list(C.keys())
  for key, val in ENV_DC.items():
    assert key not in pastKeys, f'make sure you are not duplicating keys {key}'
    C[key] = val
  return C

def parse_args():
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  tempC = parser.parse_args()
  Env = env_map[tempC.env]
  parser.set_defaults(**Env.ENV_DC)
  C = AttrDict(parser.parse_args().__dict__)
  return C
