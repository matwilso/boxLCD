#!/usr/bin/env python3
import argparse

from boxLCD.utils import AttrDict
import collections
from re import sub
import subprocess
from pathlib import Path

TIER0 = ['Dropbox', 'Bounce', 'Bounce2', 'Object2']
TIER1 = ['Urchin', 'Luxo', 'UrchinCube', 'LuxoCube', 'UrchinBall', 'LuxoBall']
ALL = TIER0 + TIER1
envs = {'all': ALL, '0': TIER0, '1': TIER1}

Arbiter = AttrDict()
Arbiter.model = 'MultiStepArbiter'
Arbiter.lr = 5e-4
Arbiter.bs = 32
Arbiter.total_itr = int(3e4)

Autoencoders = AttrDict()
Autoencoders.total_itr = int(3e4)
Autoencoders.lr = 5e-4
Autoencoders.bs = 32

BVAE = AttrDict()
BVAE.model = 'BVAE'
BVAE.hidden_size = 64
BVAE.vqK = 64
BVAE.vqD = 16
BVAE.nfilter = 16
BVAE.window = 5

RNLDA = AttrDict()
RNLDA.model = 'RNLDA'
RNLDA.hidden_size = 64
RNLDA.vqK = 64
RNLDA.vqD = 8
RNLDA.nfilter = 16
RNLDA.window = 5

autoencoders = {
  'BVAE': BVAE,
  'RNLDA': RNLDA,
}


def fstr(template):
  """https://stackoverflow.com/a/53671539/7211137
  Calls it multiple times to support nesting
  """
  while '{' in template:
    template = eval(f"f'{template}'")
  return template

def parse():
  args = parser.parse_args()
  K = AttrDict(**{key: val for key, val in args.__dict__.items()})
  K.datapath = Path(K.datapath)
  K.logdir = Path(K.logdir)
  return K

def reparse(**kwargs):
  parser.set_defaults(**kwargs)
  return parse()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('mode')
  parser.add_argument('--datapath', default='logs/trash/')
  parser.add_argument('--logdir', default='logs/trash/')
  parser.add_argument('--model')
  parser.add_argument('--envs', '-e', default='all')
  parser.add_argument('--dry', '-d', type=int, default=0)
  parser.add_argument('--log_n', type=int, default=1000)
  parser.add_argument('--lr', type=float)
  parser.add_argument('--total_itr')
  dpath = '{str(Path(K.datapath)/env)}'
  K = parse()

  TRAIN_TEMPLATE = "python -m research.main --mode=train --model={K.model} --lr={K.lr} --log_n={K.log_n} --bs={K.bs} --datapath={dpath} --logdir={logdir} --total_itr={K.total_itr}"

  if K.mode == 'collect':
    cmd_template = "python -m research.main --mode=collect --num_envs=10 --train_barrels=100 --test_barrels=10 --env={env} --logdir={dpath}"
  elif K.mode == 'Arbiter':
    K = reparse(**Arbiter)
    logdir = '{K.logdir/env}'
    cmd_template =  TRAIN_TEMPLATE + " --nfilter=64 --hidden_size=256 --window=5"
  elif K.mode == 'autoencoder':
    comb_args = {**Autoencoders, **autoencoders[K.model]}
    K = reparse(**comb_args)
    params = [f'--{key}={val}' for key, val in comb_args.items()]
    autodir = K.logdir / 'autoencoders'
    path = autodir / K.model
    logdir = '{path/env}'
    cmd_template =  TRAIN_TEMPLATE + ' '.join(params)
  elif K.mode == 'video':
    comb_args = {**Autoencoders, **autoencoders[K.model]}
    K = reparse(**comb_args)
    params = [f'--{key}={val}' for key, val in comb_args.items()]
    autodir = K.logdir / 'autoencoders'
    path = autodir / K.model
    logdir = '{path/env}'
    cmd_template =  TRAIN_TEMPLATE + ' '.join(params)

  print(K.dry)
  for env in ALL:
    cmd = fstr(cmd_template)
    if K.dry:
      print(cmd)
    else:
      subprocess.run(cmd.split(' '))

