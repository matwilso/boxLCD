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
    self.viz_idxs = np.arange(0, self.G.window * 8, self.G.window)
    self.batch_proc = lambda x: x

  def train_step(self, batch, dry=False):
    return super().train_step(self.batch_proc(batch), dry)

  # TODO: add support for different types of encoders. like the distribution. and sampling or taking the mode. or doing the logits.
  # TODO: same for decoder
  def encode(self, batch):
    raise NotImplementedError

  def decode_mode(self, z):
    mode = {}
    dists = self._decode(z)
    if 'lcd' in dists:
      mode['lcd'] = 1.0 * (dists['lcd'].probs > 0.5)
    if 'proprio' in dists:
      mode['proprio'] = dists['proprio'].mean
    if 'action' in dists:
      mode['action'] = dists['action'].mean
    return mode

  def decode_dist(self, z):
    return self._decode(z)

  def sample(self, n, mode='mode'):
    z = self.sample_z(n)
    if mode == 'mode':
      return self.decode_mode(z)
    if mode == 'dist':
      return self.decode_dist(z)

  def evaluate(self, epoch, writer, batch, arbiter=None):
    proc_batch = self.batch_proc(batch)
    metrics = {}
    self._unprompted_eval(epoch, writer, metrics, proc_batch, arbiter)
    self._prompted_eval(epoch, writer, metrics, proc_batch, arbiter)
    return metrics

  def _plot_lcds(self, epoch, writer, pred, truth=None):
    """visualize lcd reconstructions"""
    pred = pred[self.viz_idxs].cpu()
    if truth is not None:
      truth = self.unproc(truth[self.viz_idxs]).cpu()
      error = (pred - truth + 1.0) / 2.0
      stack = th.cat([truth, pred, error], -2)
      writer.add_image('recon_lcd', utils.combine_rgbs(stack, 1, 8), epoch)
    else:
      writer.add_image('sample_lcd', utils.combine_rgbs(pred, 1, 8), epoch)

  def _plot_proprios(self, epoch, writer, pred, truth=None):
    """visualize proprio reconstructions"""
    pred_proprio = pred[self.viz_idxs].cpu()
    preds = []
    for s in pred_proprio:
      preds += [self.env.reset(proprio=s)['lcd']]
    preds = 1.0 * np.stack(preds)

    if truth is not None:
      true_proprio = truth[self.viz_idxs].cpu()
      truths = []
      for s in true_proprio:
        truths += [self.env.reset(proprio=s)['lcd']]
      truths = 1.0 * np.stack(truths)
      error = (preds - truths + 1.0) / 2.0
      stack = np.concatenate([truths, preds, error], -2)[:, None]
      writer.add_image('recon_proprio', utils.combine_rgbs(stack, 1, 8), epoch)
    else:
      writer.add_image('sample_proprio', utils.combine_rgbs(preds[:, None], 1, 8), epoch)

  def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
    n = batch['lcd'].shape[0]
    decoded = self.sample(n)

    if 'lcd' in decoded:
      sample_lcd = decoded['lcd']
      self._plot_lcds(epoch, writer, sample_lcd)

    if 'proprio' in decoded:
      sample_proprio = decoded['proprio']
      self._plot_proprios(epoch, writer, sample_proprio)

    if arbiter is not None:
      decoded['lcd'] = self.proc(decoded['lcd'])
      paz = arbiter.forward(decoded).cpu().numpy()
      taz = arbiter.forward(batch).cpu().numpy()
      metrics['eval/fid'] = utils.compute_fid(paz, taz)

  def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
    # run the examples through encoder and decoder
    z = self.encode(batch, flatten=False)
    decoded = self.decode_mode(z)
    if 'lcd' in decoded:
      pred_lcd = decoded['lcd']
      true_lcd = batch['lcd']
      # run basic metrics
      self.ssim.update((pred_lcd, self.unproc(true_lcd)))
      ssim = self.ssim.compute().cpu().detach()
      metrics['eval/ssim'] = ssim
      self.psnr.update((pred_lcd, self.unproc(true_lcd)))
      psnr = self.psnr.compute().cpu().detach()
      metrics['eval/psnr'] = psnr
      # visualize reconstruction
      self._plot_lcds(epoch, writer, pred_lcd, true_lcd)

    if 'proprio' in decoded:
      pred_proprio = decoded['proprio']
      true_proprio = batch['proprio']
      metrics['eval/proprio_log_mse'] = ((true_proprio - pred_proprio)**2).mean().log().cpu()
      # visualize proprio reconstructions
      self._plot_proprios(epoch, writer, pred_proprio, true_proprio)

    if arbiter is not None:
      decoded['lcd'] = decoded['lcd'][:, 0]
      paz = arbiter.forward(decoded)
      taz = arbiter.forward(batch)
      cosdist = 1 - self.cossim(paz, taz).mean().cpu()
      metrics['eval/cosdist'] = cosdist


class SingleStepAE(Autoencoder):
  def __init__(self, env, G):
    super().__init__(env, G)
    self.batch_proc = utils.flat_batch
    self.proc = lambda x: x[:,0]
    self.unproc = lambda x: x[:,None]

class MultiStepAE(Autoencoder):
  def __init__(self, env, G):
    super().__init__(env, G)
    self.batch_proc = lambda x: x
    self.proc = lambda x: x
    self.unproc = lambda x: x

  def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
    n = batch['lcd'].shape[0]
    decoded = self.sample(n)

    if 'lcd' in decoded:
      sample_lcd = decoded['lcd']
      self._plot_lcds(epoch, writer, sample_lcd)

    if arbiter is not None:
      decoded['lcd'] = decoded['lcd'][:, 0]
      paz = arbiter.forward(decoded).cpu().numpy()
      taz = arbiter.forward(batch).cpu().numpy()
      metrics['eval/fid'] = utils.compute_fid(paz, taz)

  def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
    # run the examples through encoder and decoder
    z = self.encode(batch, flatten=False)
    decoded = self.decode_mode(z)
    if 'lcd' in decoded:
      pred_lcd = decoded['lcd']
      true_lcd = batch['lcd']
      # run basic metrics
      self.ssim.update((pred_lcd[:,:1], true_lcd[:,:1]))
      ssim = self.ssim.compute().cpu().detach()
      metrics['eval/ssim'] = ssim
      self.psnr.update((pred_lcd[:,:1], true_lcd[:,:1]))
      psnr = self.psnr.compute().cpu().detach()
      metrics['eval/psnr'] = psnr
      # visualize reconstruction
      self._plot_lcds(epoch, writer, pred_lcd[:,:3], true_lcd[:,:3])

    if 'proprio' in decoded:
      pred_proprio = decoded['proprio']
      true_proprio = batch['proprio']
      metrics['eval/proprio_log_mse'] = ((true_proprio - pred_proprio)**2).mean().log().cpu()

    if 'action' in decoded:
      pred_action = decoded['action']
      true_action = batch['action'][:,:-1]
      metrics['eval/action_log_mse'] = ((true_action - pred_action)**2).mean().log().cpu()

    if arbiter is not None:
      import ipdb; ipdb.set_trace()
      decoded['lcd'] = decoded['lcd'][:, 0]
      paz = arbiter.forward(decoded)
      taz = arbiter.forward(batch)
      cosdist = 1 - self.cossim(paz, taz).mean().cpu()
      metrics['eval/cosdist'] = cosdist
