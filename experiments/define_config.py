import subprocess
import sys
import pathlib
from boxLCD import utils
from envs.box import Box, Dropbox
from envs.wrappers import LCDEnv, NormalEnv

def env_fn(C, seed=None):
  def _make():
    if C.env == 'box':
      env = Box(C)
    elif C.env == 'dropbox':
      env = Dropbox(C)
    if C.lcd_render:
      env = LCDEnv(env)
    else:
      env = NormalEnv(env)

    env.seed(seed)
    return env
  return _make

def config():
  C = utils.AttrDict()
  # BASICS
  C.logdir = pathlib.Path('logs/')
  C.weightdir = pathlib.Path('.')
  C.buffdir = pathlib.Path('.')
  C.datapath = pathlib.Path('.')
  C.device = 'cuda' # 'cuda', 'cpu'
  C.mode = 'world'
  C.model = 'frame_token'
  C.ipython_mode = 0

  #C.data_mode = 'image'
  C.amp = 1
  C.cheap_render = 1
  C.done_n = 1000000
  C.save_n = 5
  C.full_state = 0
  C.num_vars = 0


  C.decode = 'binary'
  C.conv_io = 0
  C.collect_n = 100
  C.grad_clip = 10.0

  C.subset = 'image+state'

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

  C.env = 'box'

  # extra info that we set here for convenience and don't modify 
  C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
  C.commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
  return C

def args_type(default):
  if isinstance(default, bool): return lambda x: bool(['False', 'True'].index(x))
  if isinstance(default, int): return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
  if isinstance(default, pathlib.Path): return lambda x: pathlib.Path(x).expanduser()
  return type(default)