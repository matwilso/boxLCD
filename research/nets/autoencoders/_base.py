import numpy as np
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
from research.nets._base import Net
from research import utils

class Autoencoder(Net):
  def __init__(self, env, G):
    super().__init__(G)
    self.env = env

  def _flat_batch(self, batch):
    return {key: val.flatten(0, 1) for key, val in batch.items()}

  def train_step(self, batch, dry=False):
    super().train_step(self._flat_batch(batch), dry)

  # TODO: add support for different types of encoders. like the distribution. and sampling or taking the mode. or doing the logits.
  # TODO: same for decoder
  def encode(self, batch):
    raise NotImplementedError

  def decode(self, z):
    #if mode == 'dist':
    #  return decoded
    #elif mode == 'mode':
    #  return {'lcd': 1.0 * (decoded['lcd'].probs > 0.5), 'pstate': decoded['pstate'].mean}
    #elif mode == 'logits':
    #  return {'lcd': decoded['lcd'].logits, 'pstate': th.cat([decoded['pstate'].mean, decoded['pstate'].std], -1)}
    #return decoded
    raise NotImplementedError

  def evaluate(self, writer, batch, epoch):
    # run the examples through encoder and decoder
    slice_batch = {key: val[:8, 0] for key, val in batch.items()}
    z = self.encode(slice_batch)
    decoded = self.decode(z, mode='mode')

    # visualize lcd reconstructions
    if 'lcd' in decoded:
      pred_lcd = 1.0 * (decoded['lcd'].probs > 0.5)[:8]
      lcd = slice_batch['lcd'][:8,None]
      error = (pred_lcd - lcd + 1.0) / 2.0
      stack = th.cat([lcd, pred_lcd, error], -2)
      writer.add_image('reconstruction/image', utils.combine_imgs(stack, 1, 8)[None], epoch)
    # visualize state reconstructions
    if 'pstate' in decoded:
      pred_state = decoded['pstate'].mean.detach().cpu()
      true_state = slice_batch['pstate'].cpu()
      preds = []
      for s in pred_state:
        preds += [self.env.reset(pstate=s)['lcd']]
      truths = []
      for s in true_state:
        truths += [self.env.reset(pstate=s)['lcd']]
      preds = 1.0 * np.stack(preds)
      truths = 1.0 * np.stack(truths)
      error = (preds - truths + 1.0) / 2.0
      stack = np.concatenate([truths, preds, error], -2)[:, None]
      writer.add_image('reconstruction/pstate', utils.combine_imgs(stack, 1, 8)[None], epoch)