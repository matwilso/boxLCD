import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from boxLCD import envs
from boxLCD.utils import A, AttrDict, args_type
from define_config import env_fn, config

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
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from boxLCD.utils import A


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in config().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  C = parser.parse_args()

  env = env_fn(C)()
  N = C.collect_n
  obses = {key: np.zeros([N, C.ep_len, *val.shape], dtype=val.dtype) for key, val in env.observation_space.spaces.items()}
  acts = np.zeros([N, C.ep_len, env.action_space.shape[0]])
  pbar = tqdm(range(N))
  for i in pbar:
    start = time.time()
    obs = env.reset()
    for j in range(C.ep_len):
      act = env.action_space.sample()
      for key in obses:
        obses[key][i, j] = obs[key]
      acts[i, j] = act
      obs, rew, done, info = env.step(act)
    pbar.set_description(f'fps: {C.ep_len/(time.time()-start)}')
  os.makedirs('rollouts', exist_ok=True)
  np.savez(f'rollouts/{C.env}-{N}.npz', acts=acts, **obses)