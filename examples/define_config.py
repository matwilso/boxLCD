import subprocess
import sys
import pathlib
import boxLCD.utils
from boxLCD import envs, env_map
from boxLCD import C as boxLCD_C
from boxLCD import wrappers
from boxLCD.utils import args_type

def env_fn(C, seed=None):
  def _make():
    env = env_map[C.env](C)
    # wrap to make lcd show up in observation space
    env = wrappers.LCDEnv(env)
    env.seed(seed)
    return env
  return _make

def config():
  C = boxLCD.utils.AttrDict()
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
  for key, val in boxLCD_C.items():
    assert key not in pastKeys, f'make sure you are not duplicating keys {key}'
    C[key] = val
  return C
