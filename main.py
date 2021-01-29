import itertools
from torch.utils.tensorboard import SummaryWriter
import torch
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
    top = (top*16).astype(np.int)
    bot = (bot*16).astype(np.int)
    draw.ellipse(top.tolist()+bot.tolist(), fill=0)
    #draw.ellipse((8, 8, 11, 11), fill=0)
    # points = ((1,1), (2,1), (2,2), (1,2), (0.5,1.5))
    #points = ((100, 100), (200, 100), (200, 200), (100, 200), (50, 150))
    #draw.polygon((points), fill=200)
    image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
    image.save(f'imgs/{i:03d}.png')
    env.step(env.action_space.sample())
  import ipdb; ipdb.set_trace()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in flags().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  F = parser.parse_args()
  env = Box(F) 
  draw_it(env)

  if True:
    env = Box(F) 
    N = 100000
    obses = np.zeros([N, 200, env.observation_space.shape[0]])
    acts = np.zeros([N, 200, env.action_space.shape[0]])

    for i in range(N):
      obs = env.reset()
      for j in range(200):
        act = env.action_space.sample()
        obses[i,j] = obs
        acts[i,j] = act
        obs, rew, done, info = env.step(act)
      print(i)
    data = np.savez('test.npz', obses=obses, acts=acts)

  data = np.load('test.npz')
  obses = torch.as_tensor(data['obses'], dtype=torch.float32)
  acts = torch.as_tensor(data['acts'], dtype=torch.float32)
  N = obses.shape[0]
  # TODO: try transformer. adapt it for this scenario. and also try wavenet.
  # set it up like we have been the autoreg task. should work.
  # design tests metrics.
  from nets.transformer import Transformer
  F.block_size = 200
  model = Transformer(env, F)
  optimizer = Adam(model.parameters(), lr=F.lr)
  writer = SummaryWriter(F.logdir)
  logger = utils.dump_logger({}, writer, 0, F)

  # TODO: make sure we are sampling correctly
  # TODO: seed it to a specific starting point.

  for i in itertools.count(0):
    idxs = torch.randint(N, size=(F.bs,))
    o = obses[idxs,:].to(F.device)
    a = acts[idxs,:].to(F.device)

    optimizer.zero_grad()
    loss = model.nll(o, a)
    loss.backward()
    optimizer.step()

    logger['loss'] += [loss.detach().cpu()]
    if i % F.log_n == 0:
      imgs = []
      # TODO: have real env samples side-by-side. probably doing that dreamer error thing
      sample = model.sample(16).cpu().numpy()
      for j in range(199):
        obs = env.reset(sample[0,j+1])
        imgs += [env.render(mode='rgb_array')]
      imgs = np.stack(imgs, axis=0)[None].transpose(0, 1, -1, 2, 3)
      writer.add_video('sample', imgs, i)
      logger = utils.dump_logger(logger, writer, i, F)