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
from flags import flags, args_type
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

class RolloutDataset(Dataset):
  def __init__(self, npzfile, train=True, F=None):
    data = np.load(npzfile)
    obses = data['obses']
    acts = data['acts']
    cut = int(len(obses)*0.8)
    if train:
      self.obses = obses[:cut]
      self.acts = acts[:cut]
    else:
      self.obses = obses[cut:]
      self.acts = acts[cut:]

  def __len__(self):
    return len(self.obses)

  def __getitem__(self, idx):
    batch =  {'o': self.obses[idx], 'a': self.acts[idx]}
    return {key: torch.as_tensor(val, dtype=torch.float32) for key, val in batch.items()}

def load_ds(F):
  from torchvision import transforms
  train_dset = RolloutDataset('test.npz', train=True, F=F)
  test_dset = RolloutDataset('test.npz', train=False, F=F)
  train_loader = DataLoader(train_dset, batch_size=F.bs, shuffle=True, pin_memory=True, num_workers=2)
  test_loader = DataLoader(test_dset, batch_size=F.bs, shuffle=True, pin_memory=True, num_workers=2)
  return train_loader, test_loader

def env_fn(F, seed=None):
  def _make():
    env = Box(F)
    env.seed(seed)
    return env
  return _make

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in flags().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  F = parser.parse_args()
  env = Box(F)
  # draw_it(env)

  if False:
    env = Box(F)
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

  #data = np.load('test.npz')
  print('wait dataload')
  train_ds, test_ds = load_ds(F)

  # TODO: try transformer. adapt it for this scenario. and also try wavenet.
  # set it up like we have been the autoreg task. should work.
  # design tests metrics.
  from nets.transformer import Transformer
  F.block_size = 200
  model = Transformer(env, F)
  optimizer = Adam(model.parameters(), lr=F.lr)
  writer = SummaryWriter(F.logdir)
  logger = utils.dump_logger({}, writer, 0, F)
  seed = 0
  N = 8
  tvenv = SyncVectorEnv([env_fn(F, seed + i) for i in range(N)], F=F)  # test vector env

  # TODO: make sure we are sampling correctly
  # TODO: seed it to a specific starting point.
  for i in itertools.count():
    for batch in train_ds:
      batch = {key: val.to(F.device) for key,val in batch.items()}
      optimizer.zero_grad()
      loss = model.nll(batch)
      loss.backward()
      optimizer.step()
      logger['loss'] += [loss.detach().cpu()]

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
      for batch in test_ds:
        batch = {key: val.to(F.device) for key,val in batch.items()}
        loss = model.nll(batch)
        total_loss += loss * batch['o'].shape[0]
      avg_loss = total_loss / len(test_ds.dataset)
    logger['test/bits_per_dim'] = avg_loss.item() / np.log(2)

    # EVAL
    reset_states = np.c_[np.ones(N), np.zeros(N), np.linspace(-0.8, 0.8, N), 0.5*np.ones(N)]
    rollout = [tvenv.reset(np.arange(8), reset_states)]
    real_imgs = [tvenv.render()]
    acts = []
    for _ in range(199):
      act = tvenv.action_space.sample()
      obs = tvenv.step(act)[0]
      real_imgs += [tvenv.render()]
      rollout += [obs]
      acts += [act]
    rollout = np.stack(rollout, 1)
    acts = np.stack(acts, 1)
    prompts = rollout[:,:5]

    fake_imgs = []
    # TODO: have real env samples side-by-side. probably doing that dreamer error thing
    # TODO: also have many envs in parallel to gen stuff.
    sample, logp = model.sample(N, prompts=prompts)
    sample = sample.cpu().numpy()
    logger['sample_logp'] = logp
    for j in range(199):
      obs = tvenv.reset(np.arange(N), sample[:, j + 1])
      fake_imgs += [tvenv.render()]
    fake_imgs = np.stack(fake_imgs, axis=1).transpose(0, 1, -1, 2, 3)
    real_imgs = np.stack(real_imgs, axis=1).transpose(0, 1, -1, 2, 3)
    writer.add_video('sample', fake_imgs, i, fps=50)
    writer.add_video('true', real_imgs, i, fps=50)
    logger = utils.dump_logger(logger, writer, i, F)
    writer.flush()
    model.train()





