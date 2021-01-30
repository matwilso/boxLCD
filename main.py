from sync_vector_env import SyncVectorEnv
from runners.trainer import Trainer
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
from define_config import config, args_type
from envs.box import Box

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from utils import A
import utils

def draw_it(env):
  obs = env.reset()
  for i in range(200):
    image = Image.new("1", (16, 16))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 16, 16], fill=1)

    obj = env.dynbodies['object0']
    #xy = obs[...,(2,3)]
    pos = A[obj.position]
    print(pos)
    rad = obj.fixtures[0].shape.radius
    top = (pos - rad) / A[env.WIDTH, env.HEIGHT]
    bot = (pos + rad) / A[env.WIDTH, env.HEIGHT]
    top = (top * 16).astype(np.int)
    bot = (bot * 16).astype(np.int)
    draw.ellipse(top.tolist() + bot.tolist(), fill=0)
    #draw.ellipse((8, 8, 11, 11), fill=0)
    # points = ((1,1), (2,1), (2,2), (1,2), (0.5,1.5))
    #points = ((100, 100), (200, 100), (200, 200), (100, 200), (50, 150))
    #draw.polygon((points), fill=200)
    image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
    img = 1.0 * np.array(image).repeat(16, -1).repeat(16, -2)
    plt.imsave(f'imgs/{i:03d}.png', img, cmap='gray')
    env.step(env.action_space.sample())
  import ipdb; ipdb.set_trace()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  C = parser.parse_args()

  if False:
    env = Box(C)
    draw_it(env)
  elif False:
    env = Box(C)
    N = 100000
    obses = np.zeros([N, 200, env.observation_space.shape[0]])
    acts = np.zeros([N, 200, env.action_space.shape[0]])

    for i in range(N):
      obs = env.reset()
      for j in range(200):
        act = env.action_space.sample()
        obses[i, j] = obs
        acts[i, j] = act
        obs, rew, done, info = env.step(act)
      print(i)
    data = np.savez('test.npz', obses=obses, acts=acts)
  else:
    trainer = Trainer(C)
    trainer.run()