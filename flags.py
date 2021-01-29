import subprocess
import sys
import pathlib
import utils

def flags():
  F = utils.AttrDict()
  # BASICS
  F.logdir = pathlib.Path('logs/')
  F.weightdir = pathlib.Path('.')
  F.buffdir = pathlib.Path('.')
  F.device = 'cuda' # 'cuda', 'cpu'

  F.bs = 256
  F.lr = 1e-3
  F.n_layer = 2
  F.n_head = 4
  F.n_embed = 128
  F.min_std = 1e-4
  F.log_n = 1000

  # ENVIRONMENT
  F.special_viewer = 0
  F.dark_mode = 0
  F.use_arms = 1
  F.num_objects = 1
  F.num_agents = 0
  F.use_images = 0
  F.env_size = 64
  F.env_wh_ratio = 1.0
  F.ep_len = 200
  F.env_version = '0.7'
  F.angular_offset = 0
  F.root_offset = 0
  F.obj_offset = 0
  F.cname = 'urchin'
  F.compact_obs = 0
  F.use_speed = 1
  F.reward_mode = 'goal'
  F.only_obj_goal = 0
  F.only_model = 0
  F.move_reward = 0
  F.move_reward_weight = 0.1

  F.succ_check = 1
  F.all_contact = 1

  F.ss_model = 0
  F.model_diayn = 0
  F.model_diayn_weight = 1.0

  F.all_corners = 0
  F.use_done = 0
  F.threshold_done = 0
  F.thres = 0.05
  F.ent_rew = 1
  F.walls = 1
  F.specnorm = 0


  F.tsteps = 10
  F.step_size = 0.1
  F.manual_test = 0


  F.split_q = 0
  F.cond_layers = 0


  # extra info that we set here for convenience and don't modify 
  F.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
  F.commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
  return F

def args_type(default):
  if isinstance(default, bool): return lambda x: bool(['False', 'True'].index(x))
  if isinstance(default, int): return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
  if isinstance(default, pathlib.Path): return lambda x: pathlib.Path(x).expanduser()
  return type(default)