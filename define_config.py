import subprocess
import sys
import pathlib
import utils


def config():
  C = utils.AttrDict()
  # BASICS
  C.logdir = pathlib.Path('logs/')
  C.weightdir = pathlib.Path('.')
  C.buffdir = pathlib.Path('.')
  C.device = 'cuda' # 'cuda', 'cpu'

  C.bs = 256
  C.lr = 1e-3
  C.n_layer = 2
  C.n_head = 4
  C.n_embed = 128
  C.min_std = 1e-4
  C.log_n = 1000
  C.data_frac = 1.0
  C.vanished = 1
  C.num_envs = 5

  C.dist_head = 'gauss'
  C.mdn_k = 5
  C.dist_delta = 1

  # ENVIRONMENT
  C.special_viewer = 0
  C.dark_mode = 0
  C.use_arms = 1
  C.num_objects = 1
  C.num_agents = 0
  C.use_images = 0
  C.env_size = 128
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
  C.only_model = 0
  C.move_reward = 0
  C.move_reward_weight = 0.1

  C.succ_check = 1
  C.all_contact = 1

  C.ss_model = 0
  C.model_diayn = 0
  C.model_diayn_weight = 1.0

  C.all_corners = 0
  C.use_done = 0
  C.threshold_done = 0
  C.thres = 0.05
  C.ent_rew = 1
  C.walls = 1
  C.specnorm = 0


  C.tsteps = 10
  C.step_size = 0.1
  C.manual_test = 0


  C.split_q = 0
  C.cond_layers = 0


  # extra info that we set here for convenience and don't modify 
  C.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
  C.commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
  return C

def args_type(default):
  if isinstance(default, bool): return lambda x: bool(['False', 'True'].index(x))
  if isinstance(default, int): return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
  if isinstance(default, pathlib.Path): return lambda x: pathlib.Path(x).expanduser()
  return type(default)