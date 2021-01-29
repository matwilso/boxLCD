import pathlib
from collections import defaultdict
import yaml
from shutil import copyfile
import torch
import scipy
import re
import numpy as np

# utils
def subdict(dict, keys): return {key: dict[key] for key in keys}
def filter(dict, name): return {key: dict[key] for key in dict if re.match(name, key) is not None}
def lfilter(list, name): return [item for item in list if re.match(name, item) is not None]


def sortdict(x): return subdict(x, sorted(x))
def subdlist(dict, keys): return [dict[key] for key in keys]


def get_angle(sin, cos): return np.arctan2(sin, cos)
# map from -1,1 to bounds
def mapto(a, lowhigh): return ((a + 1.0) / (2.0) * (lowhigh[1] - lowhigh[0])) + lowhigh[0]
# map from bounds to -1,1
def rmapto(a, lowhigh): return ((a - lowhigh[0]) / (lowhigh[1] - lowhigh[0]) * (2)) + -1
def umapto(a, from_lh, to_lh): return ((a - from_lh[0]) / (from_lh[1] - from_lh[0]) * (to_lh[0] + to_lh[1])) + to_lh[0]

def tileN(x, N): return torch.tile(x[None], [N] + [1] * len(x.shape))

def combined_shape(length, shape=None):
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
  return sum([np.prod(p.shape) for p in module.parameters()])

def dump_logger(logger, writer, i, F):
  print('=' * 30)
  print(i)
  for key in logger:
    val = np.mean(logger[key])
    writer.add_scalar(key, val, i)
    print(key, val)
  print(F.full_cmd)
  with open(pathlib.Path(F.logdir) / 'hps.yaml', 'w') as f:
    yaml.dump(F, f)
  print('=' * 30)
  return defaultdict(lambda: [])

class X:
  """
  Singleton object to be able to easily create numpy arrays without having to type as much

  usage:
    >>> A = X()
    >>> arr = A[1, 2, 3]
    array([1, 2, 3])
    >>> arr = A[-1.0, 1.0]
    array([-1.0, 1.0])
  """

  def __getitem__(self, stuff):
    return np.array(stuff)


A = X()
def make_rot(angle): return A[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]

class AttrDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

class DWrap():
  """
  Dictionary Wrapper

  Gives an array an interface like a dictionary, so you can index into it by keys.
  Also handles mapping between different ranges, like [-1,1] to true bounds.

  DWrap is one way to handle named array elements. Another option is to convert back and forth between vector and dictionary.
  DWrap can be faster and more convenient.
  """

  def __init__(self, arr, arr_info, map=True):
    self.arr = arr
    self.arr_info = arr_info
    self.map = map

  def _name2idx(self, name):
    """get the index of an element in the array"""
    return self.arr_info['keys'].index(name)

  def __call__(self, key):
    return self[key]

  def __getitem__(self, key):
    """access an array by index name or list of names"""
    if isinstance(key, str):
      idx = self._name2idx(key)
      if self.map:
        return mapto(self.arr[..., idx], self.arr_info[key])
    else:
      idx = [self._name2idx(subk) for subk in key]
      if self.map:
        bounds = np.array([self.arr_info[subk] for subk in key]).T
        return mapto(self.arr[..., idx], bounds)
    return self.arr[..., idx]

  def __setitem__(self, key, item):
    """set the array indexes by name or list of names"""
    if isinstance(key, str):
      idx = self._name2idx(key)
      if self.map:
        self.arr[..., idx] = rmapto(item, self.arr_info[key])
        return
    else:
      idx = [self._name2idx(subk) for subk in key]
      if self.map:
        bounds = np.array([self.arr_info[subk] for subk in key]).T
        self.arr[..., idx] = rmapto(item, bounds)
        return
    self.arr[..., idx] = item


def write_gif(name, frames, fps=20):
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
