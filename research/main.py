from os import pipe
from tqdm import tqdm
import yaml
import time
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
from research.nets import net_map
import data

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  temp_cfg = parser.parse_args()
  # grab defaults from the env
  Env = env_map[temp_cfg.env]
  parser.set_defaults(**Env.ENV_DG)
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
  G = parser.parse_args()
  G.lcd_w = int(G.wh_ratio * G.lcd_base)
  G.lcd_h = G.lcd_base
  G.imsize = G.lcd_w * G.lcd_h
  #assert G.lcd_w == data_cfg.lcd_w and G.lcd_h == data_cfg.lcd_w, "mismatch of env dims"
  env = env_fn(G)()
  if G.mode not in ['collect']:
    if G.model in net_map:
      model = net_map[G.model](env, G)
    else:
      assert False, f"we don't have that model, {G.model} {net_map}"
    model.to(G.device)
    G.num_vars = utils.count_vars(model)

  if G.mode == 'train':
    runner = runners.Trainer(model, env, G)
  elif G.mode == 'viz':
    runner = runners.Vizer(model, env, G)
  elif G.mode == 'collect':
    data.collect(env_fn, G)
    exit()
  elif G.mode == 'fiddle':
    runner = runners.Fiddler(model, env, G)
  if G.ipython_mode:
    import IPython
    from traitlets.config import Config
    c = Config()
    c.InteractiveShellApp.exec_lines = ['runner.run()']
    c.TerminalInteractiveShell.banner2 = '***Welcome to Quick Iter Mode***'
    IPython.start_ipython(config=c, user_ns=locals())
  else:
    runner.run()