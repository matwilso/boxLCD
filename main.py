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
from envs.box import Box
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener, revoluteJointDef)

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from utils import A
import utils
from runners.video_trainer import VideoTrainer
from runners.image_trainer import ImageTrainer

def draw_it2(env):
  obs = env.reset()
  img = env.lcd_render()
  big = env.render()
  for i in range(C.ep_len):
    img = np.array(255*img, dtype=np.uint8).repeat(8, -1).repeat(8, -2)
    cat = np.concatenate([big[-128:], img[...,None].repeat(3,axis=-1)], 1)
    plt.imsave(f'imgs/{i:03d}.png', cat, cmap='gray')
    env.step(env.action_space.sample())
    img = env.lcd_render()
    big = env.render()
  import ipdb; ipdb.set_trace()

# TODO: handle partial obs for agent info, not just object
# TODO [2021/02/01]: separate out links and actions and feed in separately in an MHDPA or TF block for good GNN mixing.
# TODO [2021/02/01]: refactor all my shit and clean it up at night

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  C = parser.parse_args()

  if C.mode == 'image':
    trainer = ImageTrainer(C)
    trainer.run()
  elif C.mode == 'video':
    trainer = VideoTrainer(C)
    trainer.run()
  elif C.mode == 'lcd':
    env = Box(C)
    draw_it2(env)
  elif C.mode == 'collect':
    env = env_fn(C)()
    N = 1000
    space = env.observation_space
    obses = {key: np.zeros([N, C.ep_len, *val.shape], dtype=val.dtype) for key, val in env.observation_space.spaces.items()}
    acts = np.zeros([N, C.ep_len, env.action_space.shape[0]])
    for i in range(N):
      start = time.time()
      obs = env.reset()
      for j in range(C.ep_len):
        act = env.action_space.sample()
        for key  in obses:
          obses[key][i, j] = obs[key]
        acts[i, j] = act
        obs, rew, done, info = env.step(act)
        #env.render()
        #plt.imshow(1.0*env.lcd_render()); plt.show()
      print(f'{i} fps: {C.ep_len/(time.time()-start)}')
    lcd = '-lcd' if C.lcd_render else ''
    C.datapath.mkdir(exist_ok=True)
    data = np.savez(f'{C.datapath}/{C.env}{lcd}.npz', acts=acts, **obses)