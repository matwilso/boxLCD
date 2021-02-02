import subprocess
import sys
import pathlib
import utils
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
  C.mode = 'video'
  #C.data_mode = 'image'
  C.amp = 1

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

  C.lcd_h = 16
  C.lcd_w = 16
  C.env_size = 128
  C.lcd_render = 0 

  # ENVIRONMENT
  C.env = 'box'
  C.special_viewer = 0
  C.dark_mode = 0
  C.use_arms = 1
  C.num_objects = 1
  C.num_agents = 0
  C.use_images = 0
  C.env_wh_ratio = 1.0
  C.ep_len = 200
  C.env_version = '0.7'
  C.angular_offset = 0
  C.root_offset = 0
  C.obj_offset = 0
  C.cname = 'urchin'
  C.compact_obs = 0
  C.use_speed = 1
  C.reward_mode = 'goal'
  C.only_obj_goal = 0

  C.succ_check = 1
  C.all_contact = 1

  C.all_corners = 0
  C.use_done = 0
  C.threshold_done = 0
  C.thres = 0.05
  C.walls = 1

  # extra info that we set here for convenience and don't modify 
  C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
  C.commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
  return C

def args_type(default):
  if isinstance(default, bool): return lambda x: bool(['False', 'True'].index(x))
  if isinstance(default, int): return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
  if isinstance(default, pathlib.Path): return lambda x: pathlib.Path(x).expanduser()
  return type(default)