import numpy as np
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
from research.nets._base import Net
from research import utils
from scipy.linalg import fractional_matrix_power

class Autoencoder(Net):
  def __init__(self, env, G):
    super().__init__(G)
    self.env = env
    self.viz_idxs = np.arange(0, self.G.window * 8, self.G.window)

  def _flat_batch(self, batch):
    return {key: val.flatten(0, 1) for key, val in batch.items()}

  def train_step(self, batch, dry=False):
    return super().train_step(self._flat_batch(batch), dry)

  # TODO: add support for different types of encoders. like the distribution. and sampling or taking the mode. or doing the logits.
  # TODO: same for decoder
  def encode(self, batch):
    raise NotImplementedError

  def decode_mode(self, z):
    mode = {}
    dists = self._decode(z)
    if 'lcd' in dists:
      mode['lcd'] = 1.0 * (dists['lcd'].probs > 0.5)
    if 'pstate' in dists:
      mode['pstate'] = dists['pstate'].mean
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
    flat_batch = self._flat_batch(batch)
    metrics = {}
    self._unprompted_eval(epoch, writer, metrics, flat_batch, arbiter)
    self._prompted_eval(epoch, writer, metrics, flat_batch, arbiter)
    return metrics

  def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
    n = batch['lcd'].shape[0]
    decoded = self.sample(n)

    if 'lcd' in decoded:
      sample_lcd = decoded['lcd']
      self._plot_lcds(epoch, writer, sample_lcd)

    if 'pstate' in decoded:
      sample_pstate = decoded['pstate']
      self._plot_pstates(epoch, writer, sample_pstate)

    if arbiter is not None:
      decoded['lcd'] = decoded['lcd'][:, 0]
      paz = arbiter.forward(decoded).cpu().numpy()
      taz = arbiter.forward(batch).cpu().numpy()

      # FID / Wasserstein Computation
      # https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions
      # https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance
      # aggregate stats from this batch
      pmu = np.mean(paz, 0)
      pcov = np.cov(paz, rowvar=False)
      tmu = np.mean(taz, 0)
      tcov = np.cov(taz, rowvar=False)
      assert pcov.shape[0] == paz.shape[-1]
      # compute FID equation
      fid = np.mean((pmu - tmu)**2) + np.trace(pcov + tcov - 2 * fractional_matrix_power(pcov.dot(tcov), 0.5))
      metrics['eval/fid'] = fid

  def _plot_lcds(self, epoch, writer, pred, truth=None):
    """visualize lcd reconstructions"""
    pred = pred[self.viz_idxs].cpu()
    if truth is not None:
      truth = truth[self.viz_idxs, None].cpu()
      error = (pred - truth + 1.0) / 2.0
      stack = th.cat([truth, pred, error], -2)
      writer.add_image('recon_lcd', utils.combine_imgs(stack, 1, 8)[None], epoch)
    else:
      writer.add_image('sample_lcd', utils.combine_imgs(pred, 1, 8)[None], epoch)

  def _plot_pstates(self, epoch, writer, pred, truth=None):
    """visualize pstate reconstructions"""
    pred_pstate = pred[self.viz_idxs].cpu()
    preds = []
    for s in pred_pstate:
      preds += [self.env.reset(pstate=s)['lcd']]
    preds = 1.0 * np.stack(preds)

    if truth is not None:
      true_pstate = truth[self.viz_idxs].cpu()
      truths = []
      for s in true_pstate:
        truths += [self.env.reset(pstate=s)['lcd']]
      truths = 1.0 * np.stack(truths)
      error = (preds - truths + 1.0) / 2.0
      stack = np.concatenate([truths, preds, error], -2)[:, None]
      writer.add_image('recon_pstate', utils.combine_imgs(stack, 1, 8)[None], epoch)
    else:
      writer.add_image('sample_pstate', utils.combine_imgs(preds[:,None], 1, 8)[None], epoch)

  def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
    # run the examples through encoder and decoder
    z = self.encode(batch, flatten=False)
    decoded = self.decode_mode(z)
    if 'lcd' in decoded:
      pred_lcd = decoded['lcd']
      true_lcd = batch['lcd']
      # run basic metrics
      self.ssim.update((pred_lcd, true_lcd[:,None]))
      ssim = self.ssim.compute().cpu().detach()
      metrics['eval/ssim'] = ssim
      self.psnr.update((pred_lcd, true_lcd[:,None]))
      psnr = self.psnr.compute().cpu().detach()
      metrics['eval/psnr'] = psnr
      # visualize reconstruction
      self._plot_lcds(epoch, writer, pred_lcd, true_lcd) 

    if 'pstate' in decoded:
      pred_pstate = decoded['pstate']
      true_pstate = batch['pstate']
      metrics['eval/pstate_log_mse'] = ((true_pstate - pred_pstate)**2).mean().log().cpu()
      # visualize pstate reconstructions
      self._plot_pstates(epoch, writer, pred_pstate, true_pstate)

    if arbiter is not None:
      decoded['lcd'] = decoded['lcd'][:, 0]
      pred_az = arbiter.forward(decoded)
      true_az = arbiter.forward(batch)
      cosdist = 1 - self.cossim(pred_az, true_az).mean().cpu()
      metrics['eval/cosdist'] = cosdist