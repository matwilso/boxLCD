import argparse
import subprocess
import sys
import pathlib
import boxLCD.utils
from boxLCD import envs, env_map
from boxLCD import ENV_DC
from boxLCD.utils import args_type

def env_fn(C, seed=None):
  def _make():
    env = env_map[C.env](C)
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
  C.num_epochs = 1000000
  C.save_n = 5

  C.decode = 'multi'
  C.conv_io = 0
  C.num_barrels = 10
  C.grad_clip = 10.0

  C.bs = 64
  C.lr = 1e-4
  C.n_layer = 2
  C.n_head = 4
  C.n_embed = 128
  C.hidden_size = 128
  C.vidstack = -1
  C.stacks_per_block = 32

  C.vqD = 128
  C.vqK = 256
  C.beta = 0.25


  C.min_std = 1e-4
  C.log_n = 1
  C.data_frac = 1.0
  C.vanished = 1
  C.num_envs = 8

  C.mdn_k = 5
  C.dist_delta = 0
  C.sample_sample = 0
  C.skip_train = 0

  C.phase = 1
  C.window = 200

  C.env = 'dropbox'

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