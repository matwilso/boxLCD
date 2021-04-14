import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from boxLCD import envs, env_map
from boxLCD.utils import A, AttrDict, args_type
from utils import parse_args

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
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from boxLCD.utils import A

if __name__ == '__main__':
  G = parse_args()
  env = env_map[G.env](G)
  N = G.collect_n
  obses = {key: np.zeros([N, G.ep_len, *val.shape], dtype=val.dtype) for key, val in env.observation_space.spaces.items()}
  acts = np.zeros([N, G.ep_len, env.action_space.shape[0]])
  pbar = tqdm(range(N))
  for i in pbar:
    start = time.time()
    obs = env.reset()
    for j in range(G.ep_len):
      act = env.action_space.sample()
      for key in obses:
        obses[key][i, j] = obs[key]
      acts[i, j] = act
      obs, rew, done, info = env.step(act)
    pbar.set_description(f'fps: {G.ep_len/(time.time()-start)}')
  os.makedirs('rollouts', exist_ok=True)
  np.savez_compressed(f'rollouts/{G.env}-{N}.npz', acts=acts, **obses)