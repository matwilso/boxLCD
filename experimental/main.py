from tqdm import tqdm
import yaml
import time
from sync_vector_env import SyncVectorEnv
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import yaml
from datetime import datetime
import argparse
from define_config import config, args_type, env_fn
from boxLCD import envs
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener, revoluteJointDef)

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from boxLCD.utils import A
import utils
import runners

# TODO: handle partial obs for agent info, not just object
# TODO [2021/02/01]: separate out links and actions and feed in separately in an MHDPA or TF block for good GNN mixing.
# TODO [2021/02/09]: add some episode writer stuff. so you can write things more chunked then 1e6 rollouts.

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)

  temp_cfg = parser.parse_args()
  data_yaml = temp_cfg.datapath / 'hps.yaml'
  weight_yaml = temp_cfg.weightdir / 'hps.yaml'
  defaults = {}
  ignore = ['logdir', 'full_cmd', 'dark_mode', 'ipython_mode', 'weightdir']
  if data_yaml.exists():
    with data_yaml.open('r') as f:
      load_cfg = yaml.load(f, Loader=yaml.Loader)
    for key in load_cfg.__dict__.keys():
      if key in ignore: continue
      defaults[key] = load_cfg.__dict__[key]
  if weight_yaml.exists():
    with weight_yaml.open('r') as f:
      weight_cfg = yaml.load(f, Loader=yaml.Loader)
    for key in weight_cfg.__dict__.keys():
      if key in ignore: continue
      defaults[key] = weight_cfg.__dict__[key]
  parser.set_defaults(**defaults)
  C = parser.parse_args()
  C.lcd_w = int(C.wh_ratio*C.lcd_base)
  C.lcd_h = C.lcd_base

  if C.mode == 'world':
    trainer = runners.WorldTrainer(C)
    trainer.run()
  elif C.mode == 'viz':
    vizer = runners.Vizer(C)
    if C.ipython_mode:
      import IPython
      from traitlets.config import Config
      c = Config()
      c.InteractiveShellApp.exec_lines = ['vizer.run()']
      c.TerminalInteractiveShell.banner2 = '***Welcome to Quick Iter Mode***'
      IPython.start_ipython(config=c, user_ns=locals())
    vizer.run()
  elif C.mode == 'collect':
    env = env_fn(C)()
    N = C.collect_n
    space = env.observation_space
    obses = {key: np.zeros([N, C.ep_len, *val.shape], dtype=val.dtype) for key, val in env.observation_space.spaces.items()}
    acts = np.zeros([N, C.ep_len, env.action_space.shape[0]])
    pbar = tqdm(range(N))
    for i in pbar:
      start = time.time()
      obs = env.reset()
      for j in range(C.ep_len):
        act = env.action_space.sample()
        for key  in obses:
          obses[key][i, j] = obs[key]
        acts[i, j] = act
        obs, rew, done, info = env.step(act)
        #plt.imshow(obs['lcd']);plt.show()
        #env.render()
        #plt.imshow(1.0*env.lcd_render()); plt.show()
      pbar.set_description(f'fps: {C.ep_len/(time.time()-start)}')
    C.logdir.mkdir(parents=True, exist_ok=True)
    data = np.savez(f'{C.logdir}/dump.npz', acts=acts, **obses)
    utils.dump_logger({}, None, 0, C)