from research.nets.multistep import Multistep
from tqdm import tqdm
import yaml
import time
from sync_vector_env import SyncVectorEnv
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
import torch as th
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import yaml
from datetime import datetime
import argparse
from define_config import config, args_type, env_fn
from boxLCD import envs
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener, revoluteJointDef)

from boxLCD import env_map
import utils
import runners
from nets.combined import Combined
from nets.flatimage import FlatImageTransformer
from nets.vae import VAE
import data

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  temp_cfg = parser.parse_args()
  # grab defaults from the env
  Env = env_map[temp_cfg.env]
  parser.set_defaults(**Env.ENV_DC)
  data_yaml = temp_cfg.datapath / 'hps.yaml'
  weight_yaml = temp_cfg.weightdir / 'hps.yaml'
  defaults = {
      'vidstack': temp_cfg.ep_len,
  }
  ignore = ['logdir', 'full_cmd', 'dark_mode', 'ipython_mode', 'weightdir']
  if data_yaml.exists():
    with data_yaml.open('r') as f:
      data_cfg = yaml.load(f, Loader=yaml.Loader)
    for key in data_cfg.__dict__.keys():
      if key in ignore:
        continue
      defaults[key] = data_cfg.__dict__[key]
  if weight_yaml.exists():
    with weight_yaml.open('r') as f:
      weight_cfg = yaml.load(f, Loader=yaml.Loader)
    for key in weight_cfg.__dict__.keys():
      if key in ignore:
        continue
      defaults[key] = weight_cfg.__dict__[key]
  parser.set_defaults(**defaults)
  C = parser.parse_args()
  C.lcd_w = int(C.wh_ratio * C.lcd_base)
  C.lcd_h = C.lcd_base
  C.imsize = C.lcd_w * C.lcd_h
  #assert C.lcd_w == data_cfg.lcd_w and C.lcd_h == data_cfg.lcd_w, "mismatch of env dims"
  env = env_fn(C)()
  if C.mode not in ['collect']:
    if C.model == 'frame_token':
      model = FlatImageTransformer(env, C)
    elif C.model == 'single':
      assert C.datamode == 'image'
      model = Combined(env, C)
    elif C.model == 'multistep':
      assert C.vidstack < C.ep_len
      model = Multistep(env, C)
    elif C.model == 'vae':
      model = VAE(env, C)
    model.to(C.device)
    C.num_vars = utils.count_vars(model)

  if C.mode == 'train':
    trainer = runners.Trainer(model, env, C)
    trainer.run()
  elif C.mode == 'viz':
    vizer = runners.Vizer(model, env, C)
    if C.ipython_mode:
      import IPython
      from traitlets.config import Config
      c = Config()
      c.InteractiveShellApp.exec_lines = ['vizer.run()']
      c.TerminalInteractiveShell.banner2 = '***Welcome to Quick Iter Mode***'
      IPython.start_ipython(config=c, user_ns=locals())
    vizer.run()
  elif C.mode == 'collect':
    data.collect(env_fn(C), C)
