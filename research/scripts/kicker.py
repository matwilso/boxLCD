#!/usr/bin/env python3
import argparse
from collections import defaultdict

from matplotlib.pyplot import tricontour

from boxLCD.utils import AttrDict
from re import sub
import subprocess
from pathlib import Path

TIER0 = ['Dropbox', 'Bounce', 'Bounce2', 'Object2']
TIER1 = ['Urchin', 'Luxo', 'UrchinCube', 'LuxoCube', 'UrchinBall', 'LuxoBall']
ALL = TIER0 + TIER1
envs = {'all': ALL, '0': TIER0, '1': TIER1}

MultiStepArbiter = AttrDict()
MultiStepArbiter.total_itr = int(3e4)

Encoder = AttrDict()
Encoder.total_itr = int(3e4)

BVAE = AttrDict()
BVAE.hidden_size = 64
BVAE.vqK = 64
BVAE.vqD = 16
BVAE.nfilter = 16
BVAE.window = 5

RNLDA = AttrDict()
RNLDA.hidden_size = 64
RNLDA.vqK = 64
RNLDA.vqD = 8
RNLDA.nfilter = 16
RNLDA.window = 5

encoder = {
    'BVAE': BVAE,
    'RNLDA': RNLDA,
}

ENV_WINDOW = defaultdict(lambda: 50)
ENV_WINDOW['Dropbox'] = 25
ENV_PROMPT = defaultdict(lambda: 3)
ENV_PROMPT['Dropbox'] = 1

Video = AttrDict()
Video.total_itr = int(1e5)
#Video.window = '{ENV_WINDOW[{env}]}'
#Video.prompt = '{ENV_PROMPT[{env}]}'
Video.arbiterdir = '{K.arbiterdir/env}'

RSSM = AttrDict()
RSSM.nfilter = 64
RSSM.hidden_size = 300
RSSM.free_nats = 0.01

FIT = AttrDict()
FIT.n_layer = 2
FIT.n_head = 4
FIT.n_embed = 256
FIT.hidden_size = 256

FBT = AttrDict()
FBT.n_layer = 4
FBT.n_head = 8
FBT.n_embed = 512
FBT.hidden_size = 512
FBT.weightdir = '{K.encoderdir/"BVAE"}/{env}'

FRNLD = AttrDict()
FRNLD.n_layer = 4
FRNLD.n_head = 8
FRNLD.n_embed = 512
FRNLD.hidden_size = 512
FRNLD.weightdir = '{K.encoderdir/"RNDLA"}/{env}'


video = {
    'RSSM': RSSM,
    'FIT': FIT,
    'FBT': FBT,
    'FRNLD': FRNLD,
}

def fstr(template):
  """https://stackoverflow.com/a/53671539/7211137
  Calls it multiple times to support nesting
  """
  while '{' in template:
    template = eval(f"f'{template}'")
  return template

def parse():
  args, _ = parser.parse_known_args()
  K = AttrDict(**{key: val for key, val in args.__dict__.items()})
  K.datadir = Path(K.datadir)
  K.logdir = Path(K.logdir)
  K.arbiterdir = Path(K.arbiterdir)
  K.encoderdir = Path(K.encoderdir)
  return K

def reparse(K, **kwargs):
  kwargs = {key: val for key, val in kwargs.items() if key not in ['arbiterdir', 'encoderdir', 'weightdir']}
  for key, val in kwargs.items():
    if key not in K: parser.add_argument(f'--{key}', default=val)
  parser.set_defaults(**kwargs)
  return parse()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('mode')
  parser.add_argument('--datadir', default='logs/trash/')
  parser.add_argument('--logdir', default='logs/trash/')
  parser.add_argument('--arbiterdir', default='logs/trash/')
  parser.add_argument('--encoderdir', default='logs/trash/')
  parser.add_argument('--model')
  parser.add_argument('--envs', '-e', default='all')
  parser.add_argument('--dry', '-d', type=int, default=0)
  parser.add_argument('--log_n', type=int, default=1000)
  parser.add_argument('--lr', type=float, default=5e-4)
  parser.add_argument('--bs', type=float, default=32)
  parser.add_argument('--total_itr')
  ddir = '{str(Path(K.datadir)/env)}'
  K = parse()

  TRAIN_TEMPLATE = "python -m research.main --mode=train --model={K.model} --lr={K.lr} --bs={K.bs} --log_n={K.log_n} --datadir={ddir} --logdir={logdir} --total_itr={K.total_itr} "

  if K.mode == 'collect':
    cmd_template = "python -m research.main --mode=collect --num_envs=10 --train_barrels=100 --test_barrels=10 --env={env} --logdir={ddir}"
  elif K.mode == 'arbiter':
    K = reparse(K, **MultiStepArbiter)
    logdir = '{K.arbiterdir/env}'
    cmd_template = TRAIN_TEMPLATE + " --nfilter=64 --hidden_size=256 --window=5"
  elif K.mode == 'train':
    if K.model in encoder:
      comb_args = {**Encoder, **encoder[K.model]}
      dir = K.logdir / 'encoder'
    elif K.model in video:
      comb_args = {**Video, **video[K.model]}
      dir = K.logdir / 'video'
    else:
      raise Exception('wrong model')
    K = reparse(K, **comb_args)
    params = [f'--{key}={val}' for key, val in comb_args.items()]
    path = dir / K.model
    logdir = '{path/env}'
    cmd_template = TRAIN_TEMPLATE + ' '.join(params)

  if K.dry: print('DRY RUN')
  for env in ALL:
    cmd = fstr(cmd_template)
    if K.dry:
      print(cmd)
    else:
      subprocess.run(cmd.split(' '))