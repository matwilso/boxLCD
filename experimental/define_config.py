import subprocess
import sys
import pathlib
import boxLCD.utils
from boxLCD import envs
from boxLCD import C as boxLCD_C
from boxLCD import wrappers
from boxLCD.utils import args_type

def env_fn(C, seed=None):
  def _make():
    if C.env == 'dropbox':
      env = envs.Dropbox(C)
    elif C.env == 'bounce':
      env = envs.Bounce(C)
    elif C.env == 'boxor':
      env = envs.BoxOrCircle(C)
    elif C.env == 'urchin':
      env = envs.Urchin(C)
    elif C.env == 'urchin_ball':
      env = envs.UrchinBall(C)
    elif C.env == 'urchin_balls':
      env = envs.UrchinBalls(C)
    elif C.env == 'urchin_cubes':
      env = envs.UrchinCubes(C)
    env = wrappers.LCDEnv(env)
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
  C.mode = 'world'
  C.model = 'frame_token'
  C.ipython_mode = 0

  #C.data_mode = 'image'
  C.amp = 1
  C.done_n = 1000000
  C.save_n = 5
  C.full_state = 0
  C.num_vars = 0

  C.decode = 'multi'
  C.conv_io = 0
  C.collect_n = 100
  C.grad_clip = 10.0

  C.subset = 'image'

  C.bs = 64
  C.lr = 1e-4
  C.n_layer = 2
  C.n_head = 4
  C.n_embed = 128
  C.min_std = 1e-4
  C.log_n = 1000
  C.data_frac = 1.0
  C.vanished = 1
  C.num_envs = 5

  C.mdn_k = 5
  C.dist_delta = 0
  C.sample_sample = 0
  C.skip_train = 0

  C.env = 'dropbox'

  # extra info that we set here for convenience and don't modify 
  C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
  C.commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')

  pastKeys = list(C.keys())
  for key, val in boxLCD_C.items():
    assert key not in pastKeys, f'make sure you are not duplicating keys {key}'
    C[key] = val
  return C