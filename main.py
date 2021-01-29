import numpy as np
import yaml
from datetime import datetime
import argparse
from flags import flags, args_type
from envs.box import Box

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in flags().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  F = parser.parse_args()
  if True:
    env = Box(F) 
    N = 10000
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
  obses = obses
  acts = acts
  # TODO: try transformer. adapt it for this scenario. and also try wavenet.
  # set it up like we have been the autoreg task. should work.
  # design tests metrics.