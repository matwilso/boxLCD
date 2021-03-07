import time
import pathlib
from collections import defaultdict
import yaml
from shutil import copyfile
import torch
import scipy
import re
import numpy as np

def tileN(x, N): return torch.tile(x[None], [N] + [1] * len(x.shape))

def combined_shape(length, shape=None):
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
  return sum([np.prod(p.shape) for p in module.parameters()])

def zero_module(module):
  """Zero out the parameters of a module and return it."""
  for p in module.parameters():
    p.detach().zero_()
  return module

def dump_logger(logger, writer, i, C):
  print('=' * 30)
  print(i)
  for key in logger:
    val = np.mean(logger[key])
    if writer is not None:
      writer.add_scalar(key, val, i)
      if key == 'loss' and i > 0:
        writer.add_scalar('logx/' + key, val, int(np.log(1e5 * i)))
      # if 'loss' in key:
      #  writer.add_scalar('neg/'+key, -val, i)
    print(key, val)
  print(C.full_cmd)
  print(C.num_vars)
  with open(pathlib.Path(C.logdir) / 'hps.yaml', 'w') as f:
    yaml.dump(C, f)
  print('=' * 30)
  return defaultdict(lambda: [])

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
    out = torch.cat([out, torch.zeros(out.shape[:-1])[..., None]], -1)
  out = out.reshape(T, C, H, N * (W + 1))[None]
  return out

def combine_imgs(arr, row=5, col=5):
  """takes batch of video or image and pushes the batch dim into certain image shapes given by b,row,col"""
  if len(arr.shape) == 4:  # image
    BS, C, H, W = arr.shape
    assert BS == row * col, (BS, row, col, H, W)
    if isinstance(arr, np.ndarray):
      x = arr.reshape([row, col, H, W]).transpose(0, 2, 1, 3).reshape([row * H, col * W])
    else:
      x = arr.reshape([row, col, H, W]).permute(0, 2, 1, 3).flatten(0, 1).flatten(-2)
    return x
  elif len(arr.shape) == 5:  # video
    BS, T, C, H, W = arr.shape
    assert BS == row * col, (BS, T, row, col, H, W)
    x = arr.reshape([row, col, T, H, W]).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2)
    return x
  else:
    assert False, (arr.shape, arr.ndim)