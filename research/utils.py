import os
import subprocess
import time
import pathlib
from collections import defaultdict
import yaml
from shutil import copyfile
import torch as th
import scipy
import re
import numpy as np
from scipy.linalg import fractional_matrix_power
from torch import nn

class ConciseNumpyArray:
  """
  Singleton object to be able to easily create numpy arrays without having to type as much

  usage:
    >>> A = ConciseNumpyArray()
    >>> arr = A[1, 2, 3]
    array([1, 2, 3])
    >>> arr = A[-1.0, 1.0]
    array([-1.0, 1.0])
  """
  def __getitem__(self, stuff):
    return np.array(stuff)
A = ConciseNumpyArray()

# general dictionary and list utils
def subdict(dict, subkeys): return {key: dict[key] for key in subkeys}
def sortdict(x): return subdict(x, sorted(x))
def subdlist(dict, subkeys): return [dict[key] for key in subkeys]
# filter or negative filter
def filtdict(dict, phrase, fkey=lambda x: x, fval=lambda x: x):
  return {fkey(key): fval(dict[key]) for key in dict if re.match(phrase, key) is not None}
def nfiltdict(dict, phrase): return {key: dict[key] for key in dict if re.match(phrase, key) is None}
def filtlist(list, phrase): return [item for item in list if re.match(phrase, item) is not None]
def nfiltlist(list, phrase): return [item for item in list if re.match(phrase, item) is None]
# env specific stuff
def get_angle(sin, cos): return np.arctan2(sin, cos)
def make_rot(angle): return A[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
# map from -1,1 to bounds
def mapto(a, lowhigh): return ((a + 1.0) / (2.0) * (lowhigh[1] - lowhigh[0])) + lowhigh[0]
# map from bounds to -1,1
def rmapto(a, lowhigh): return ((a - lowhigh[0]) / (lowhigh[1] - lowhigh[0]) * (2)) + -1
def prefix_dict(name, dict): return {name + key: dict[key] for key in dict}

class AttrDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

def tileN(x, N): return th.tile(x[None], [N] + [1] * len(x.shape))

def combined_shape(length, shape=None):
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
  return sum([np.prod(p.shape) for p in module.parameters()])

def dump_logger(logger, writer, i, G, verbose=True):
  print('=' * 30)
  print(i)
  for key in logger:
    check = logger[key][0] if isinstance(logger[key], list) else logger[key]
    if th.is_tensor(check):
      assert check.device.type == 'cpu', f'all metrics should be on the cpu before logging. {key} is on {check.device}'
    print(key, end=' ')
    val = np.mean(logger[key])
    if writer is not None:
      writer.add_scalar(key, val, i)
      if key == 'loss' and i > 0:
        writer.add_scalar('logx/' + key, val, int(np.log(1e5 * i)))
      # if 'loss' in key:
      #  writer.add_scalar('neg/'+key, -val, i)
    print(val)
  print(G.full_cmd)
  print(G.num_vars)
  print(G.logdir)
  with open(pathlib.Path(G.logdir) / 'hps.yaml', 'w') as f:
    yaml.dump(G, f, width=1000)
  print('=' * 30)
  return defaultdict(lambda: [])

def write_gif(name, frames, fps=30):
  start = time.time()
  from moviepy.editor import ImageSequenceClip
  # make the moviepy clip
  clip = ImageSequenceClip(list(frames), fps=fps)
  clip.write_gif(name, fps=fps)
  copyfile(name, str(pathlib.Path(f'~/Desktop/{name}').expanduser()))
  print(time.time() - start)

def write_video(name, frames, fps=20):
  start = time.time()
  import cv2
  if name.endswith('mp4'):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  else:
    import ipdb; ipdb.set_trace()
  writer = cv2.VideoWriter(name, fourcc, fps, (frames.shape[-2], frames.shape[-3]))
  for frame in frames:
    writer.write(frame[..., ::-1])
  writer.release()
  copyfile(name, str(pathlib.Path(f'~/Desktop/{name}').expanduser()))
  # print(time.time()-start)

def force_shape(out):
  """take one right before video and force it's shape"""
  #out = np.array(out)
  # TODO: add borders around and between images for easier viz
  N, T, C, H, W = out.shape
  if isinstance(out, np.ndarray):
    out = out.transpose(1, 2, 3, 0, 4)
    out = np.concatenate([out, np.zeros(out.shape[:-1], dtype=out.dtype)[..., None]], -1)
  else:
    out = out.permute(1, 2, 3, 0, 4)
    out = th.cat([out, th.zeros(out.shape[:-1])[..., None]], -1)
  out = out.reshape(T, C, H, N * (W + 1))[None]
  return out

def combine_rgbs(arr, row=5, col=5):
  """takes batch of video or image and pushes the batch dim into certain image shapes given by b,row,col"""
  if isinstance(arr, np.ndarray):
    permute = np.transpose
  else:
    def permute(x, y): return x.permute(y)

  if len(arr.shape) == 4:  # image
    BS, C, H, W = arr.shape
    assert BS == row * col, f'{(BS, row, col, H, W)} {row*col},{BS}'
    x = permute(arr.reshape([row, col, C, H, W]), (2, 0, 3, 1, 4)).reshape([C, row * H, col * W])
    return x
  elif len(arr.shape) == 5:  # video
    BS, T, C, H, W = arr.shape
    assert BS == row * col, (BS, T, row, col, C, H, W)
    x = permute(arr.reshape([row, col, T, C, H, W]), (2, 3, 0, 4, 1, 5)).reshape([T, C, row * H, col * W])
    return x
  else:
    assert False, (arr.shape, arr.ndim)


def combine_imgs(arr, row=5, col=5):
  """takes batch of video or image and pushes the batch dim into certain image shapes given by b,row,col"""
  if len(arr.shape) == 4:  # image
    BS, C, H, W = arr.shape
    assert BS == row * col, f'{(BS, row, col, H, W)} {row*col},{BS}'
    if isinstance(arr, np.ndarray):
      x = arr.reshape([row, col, H, W]).transpose(0, 2, 1, 3).reshape([row * H, col * W])
    else:
      x = arr.reshape([row, col, H, W]).permute(0, 2, 1, 3).flatten(0, 1).flatten(-2)
    return x
  elif len(arr.shape) == 5:  # video
    BS, T, C, H, W = arr.shape
    assert BS == row * col, (BS, T, row, col, H, W)
    if isinstance(arr, np.ndarray):
      x = arr.reshape([row, col, T, H, W]).transpose(2, 0, 3, 1, 4).reshape([T, row * H, col * W])
    else:
      x = arr.reshape([row, col, T, H, W]).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2)
    return x
  else:
    assert False, (arr.shape, arr.ndim)

class PTimer:
  def __init__(self, message):
    self.message = message

  def __enter__(self):
    self.time_start = time.time()

  def __exit__(self, exc_type, exc_val, exc_tb):
    new_time = time.time() - self.time_start
    print(self.message, new_time)

  def start(self):
    self.__enter__()

  def stop(self):
    self.__exit__(None, None, None)


class Timer:
  def __init__(self, logger, message):
    self.logger = logger
    self.message = message

  def __enter__(self):
    self.time_start = time.time()

  def __exit__(self, exc_type, exc_val, exc_tb):
    new_time = time.time() - self.time_start
    self.logger['dt/' + self.message] += [new_time]

  def start(self):
    self.__enter__()

  def stop(self):
    self.__exit__(None, None, None)


def add_video(writer, tag, vid_tensor, global_step=None, fps=4, walltime=None):
  th._C._log_api_usage_once("tensorboard.logging.add_video")
  from torch.utils.tensorboard import _convert_np, _utils, summary
  from tensorboard.compat.proto.summary_pb2 import Summary
  tensor = _convert_np.make_np(vid_tensor)
  tensor = _utils._prepare_video(tensor)
  scale_factor = summary._calc_scale_factor(tensor)
  tensor = tensor.astype(np.float32)
  tensor = (tensor * scale_factor).astype(np.uint8)
  video = make_video(tensor, fps)
  summ = Summary(value=[Summary.Value(tag=tag, image=video)])
  writer._get_file_writer().add_summary(summ, global_step, walltime)

def make_video(tensor, fps):
  from tensorboard.compat.proto.summary_pb2 import Summary
  try:
    import moviepy  # noqa: F401
  except ImportError:
    print('add_video needs package moviepy')
    return
  try:
    from moviepy import editor as mpy
  except ImportError:
    print("moviepy is installed, but can't import moviepy.editor. Some packages could be missing [imageio, requests]")
    return
  import tempfile
  t, h, w, c = tensor.shape
  # encode sequence of images into gif string
  clip = mpy.ImageSequenceClip(list(tensor), fps=fps)
  filename = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
  try:  # newer version of moviepy use logger instead of progress_bar argument.
    clip.write_gif(filename, verbose=False, logger=None)
  except TypeError:
    try:  # older version of moviepy does not support progress_bar argument.
      clip.write_gif(filename, verbose=False, progress_bar=False)
    except TypeError:
      clip.write_gif(filename, verbose=False)
  #subprocess.run(['gifsicle', '--lossy=30', '-o', filename, filename])
  with open(filename, 'rb') as f:
    tensor_string = f.read()
  try:
    os.remove(filename)
  except OSError:
    logging.warning('The temporary file used by moviepy cannot be deleted.')
  return Summary.Image(height=h, width=w, colorspace=c, encoded_image_string=tensor_string)

def compute_grad_norm(parameters):
  if isinstance(parameters, th.Tensor):
    parameters = [parameters]
  parameters = [p for p in parameters if p.grad is not None]
  if len(parameters) == 0:
    return th.tensor(0.)
  device = parameters[0].grad.device
  total_norm = th.norm(th.stack([th.norm(p.grad.detach()).to(device) for p in parameters]))
  return total_norm

def compute_fid(x, y):
  """
  FID / Wasserstein Computation
  https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions
  https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance
  """
  try:
    assert x.ndim == 2 and y.ndim == 2
    # aggregate stats from this batch
    pmu = np.mean(x, 0)
    pcov = np.cov(x, rowvar=False)
    tmu = np.mean(y, 0)
    tcov = np.cov(y, rowvar=False)
    assert pcov.shape[0] == x.shape[-1]
    # compute FID equation
    fid = np.mean((pmu - tmu)**2) + np.trace(pcov + tcov - 2 * fractional_matrix_power(pcov.dot(tcov), 0.5))
    # TODO: this is somewhat concerning i got an imaginary number before.
    return fid.real
  except:
    return np.nan

def flat_batch(batch):
  return {key: val.flatten(0, 1) for key, val in batch.items()}

def flatten_first(arr):
  shape = arr.shape
  return arr.reshape([shape[0] * shape[1], *shape[2:]])


def manifold_estimate(set_a, set_b, k=3):
  """https://arxiv.org/abs/1904.06991"""
  # compute manifold
  # with PTimer('cdist1'):
  d = th.cdist(set_a, set_a)
  # with PTimer('topk'):
  radii = th.topk(d, k + 1, largest=False)[0][..., -1:]
  # eval
  # with PTimer('cdist2'):
  d2 = th.cdist(set_a, set_b)
  return (d2 < radii).any(0).float().mean()

def precision_recall_f1(real, gen, k=3):
  """
  precision = realistic. fraction of generated images that are realistic
  recall = coverage. fraction of data manifold covered by generator 

  how often do you generate something that is closer to a real world sample.
  how often is there a real sample that is closer to your generated one than other generated ones are.

  real: (NxZ)
  gen: (NxZ)
  """
  precision = manifold_estimate(real, gen, k)
  recall = manifold_estimate(gen, real, k)
  f1 = 2 * (precision * recall) / (precision + recall)
  return precision, recall, f1

class Reshape(nn.Module):
  def __init__(self, *args):
    super().__init__()
    self.args = args
  def forward(self, x):
    return x.reshape(*self.args)

def discount_cumsum(x, discount):
  """
  magic from rllab for computing discounted cumulative sums of vectors.
  input: 
      vector x, 
      [x0, 
       x1, 
       x2]
  output:
      [x0 + discount * x1 + discount^2 * x2,  
       x1 + discount * x2,
       x2]
  """
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


if __name__ == '__main__':
  real = th.rand(1000, 128)
  gen = th.randn(1000, 128)
  dist = th.cdist(real, gen)
  # precision = realistic, recall = coverage.
  #precision = manifold_estimate(real, gen, 3)
  #recall = manifold_estimate(gen, real, 3)
  #f1 = 2 * (precision * recall) / (precision + recall)
  #print(precision, recall, f1)
