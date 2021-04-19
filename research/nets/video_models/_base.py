from research.wrappers import preproc_vec_env
import numpy as np
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
from research.nets._base import Net
from research import utils
from wrappers.async_vector_env import AsyncVectorEnv
from research.define_config import env_fn
from jax.tree_util import tree_map

class VideoModel(Net):
  def __init__(self, env, G):
    super().__init__(G)
    self.env = env
    self.act_n = env.action_space.shape[0]
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    self.proprio_n = env.observation_space.spaces['proprio'].shape[0]
    self.venv = AsyncVectorEnv([env_fn(G) for _ in range(8)])

  def onestep(self):
    raise NotImplementedError

  def sample(self, n, action=None, prompts=None, prompt_n=8):
    raise NotImplementedError

  def evaluate(self, epoch, writer, batch, arbiter=None):
    metrics = {}
    self._unprompted_eval(epoch, writer, metrics, batch, arbiter)
    self._prompted_eval(epoch, writer, metrics, batch, arbiter)
    self._duplicate_eval(epoch, writer, metrics, batch, arbiter)
    metrics = tree_map(lambda x: th.as_tensor(x).cpu(), metrics)
    return metrics

  def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
    n = batch['lcd'].shape[0]
    action = (th.rand(n, self.G.window, self.env.action_space.shape[0]) * 2 - 1).to(self.G.device)
    sample = self.sample(n, action)

    if 'lcd' in sample:
      self._lcd_video(epoch, writer, sample['lcd'])

    if 'proprio' in sample:
      self._proprio_video(epoch, writer, sample['proprio'])

    if arbiter is not None:
      t10 = tree_map(lambda x: x[:, 8:], batch)
      s10 = tree_map(lambda x: x[:, 8:], sample)
      s10['lcd'] = s10['lcd'][:,:,0]
      if arbiter.original_name == 'Dummy':
        s4 = tree_map(lambda x: x.reshape([-1, 4, *x.shape[2:]]), s10)
        t4 = tree_map(lambda x: x.reshape([-1, 4, *x.shape[2:]]), t10)
        sact = action[:,8:]
        tact = batch['action'][:,8:]
        sact = sact.reshape([-1, 4, *sact.shape[2:]])[:,:-1]
        tact = tact.reshape([-1, 4, *tact.shape[2:]])[:,:-1]

        paz, paa = arbiter.forward(s4)
        metrics['eval/unprompted_action_log_mse'] = ((sact - paa)**2).mean().log()
        taz, taa = arbiter.forward(t4)
        fid = utils.compute_fid(paz.cpu().numpy(), taz.cpu().numpy())
        metrics['eval/unprompted_fid'] = fid
        precision, recall, f1 = utils.precision_recall_f1(taz, paz)
        metrics['eval/unprompted_precision'] = precision.cpu()
        metrics['eval/unprompted_recall'] = recall.cpu()
        metrics['eval/unprompted_f1'] = f1.cpu()
      else:
        paz = arbiter.forward(utils.flat_batch(s10)).cpu().numpy()
        taz = arbiter.forward(utils.flat_batch(t10)).cpu().numpy()
        fid = utils.compute_fid(paz, taz)
        metrics['eval/fid'] = fid



  def _duplicate_eval(self, epoch, writer, metrics, batch, arbiter=None):
    n = batch['lcd'].shape[0]
    batch = {key: val[:1].repeat_interleave(8,0) for key, val in batch.items()}
    sample = self.sample(n, action=batch['action'], prompts=batch, prompt_n=8)
    if 'lcd' in sample:
      pred_lcd = sample['lcd']
      true_lcd = batch['lcd'][:, :, None]
      self._lcd_video(epoch, writer, pred_lcd, true_lcd, name='duplicate')
    if 'proprio' in sample:
      pred_proprio = sample['proprio']
      true_proprio = batch['proprio']
      self._proprio_video(epoch, writer, pred_proprio, true_proprio, name='duplicate')

  def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
    n = batch['lcd'].shape[0]
    sample = self.sample(n, action=batch['action'], prompts=batch, prompt_n=8)

    if 'lcd' in sample:
      pred_lcd = sample['lcd']
      true_lcd = batch['lcd'][:, :, None]
      # run basic metrics
      self.ssim.update((pred_lcd.flatten(0, 1), true_lcd.flatten(0, 1)))
      ssim = self.ssim.compute().cpu().detach()
      metrics['eval/ssim'] = ssim
      # TODO: try not flat
      self.psnr.update((pred_lcd.flatten(0, 1), true_lcd.flatten(0, 1)))
      psnr = self.psnr.compute().cpu().detach()
      metrics['eval/psnr'] = psnr
      # visualize reconstruction
      self._lcd_video(epoch, writer, pred_lcd, true_lcd)

    if 'proprio' in sample:
      pred_proprio = sample['proprio']
      true_proprio = batch['proprio']
      metrics['eval/proprio_log_mse'] = ((true_proprio - pred_proprio)**2).mean().log().cpu()
      # visualize reconstruction
      self._proprio_video(epoch, writer, pred_proprio, true_proprio)

    if arbiter is not None:
      t10 = tree_map(lambda x: x[:, 8:], batch)
      s10 = tree_map(lambda x: x[:, 8:], sample)
      s10['lcd'] = s10['lcd'][:,:,0]
      if arbiter.original_name == 'Dummy':
        s4 = tree_map(lambda x: x.reshape([-1, 4, *x.shape[2:]]), s10)
        t4 = tree_map(lambda x: x.reshape([-1, 4, *x.shape[2:]]), t10)
        tact = batch['action'][:,8:]
        tact = tact.reshape([-1, 4, *tact.shape[2:]])[:,:-1]

        paz, paa = arbiter.forward(s4)
        metrics['eval/prompted_action_log_mse'] = ((tact - paa)**2).mean().log()
        taz, taa = arbiter.forward(t4)
        metrics['eval/true_action_log_mse'] = ((tact - taa)**2).mean().log()
        fid = utils.compute_fid(paz.cpu().numpy(), taz.cpu().numpy())
        metrics['eval/prompted_fid'] = fid
        precision, recall, f1 = utils.precision_recall_f1(taz, paz)
        metrics['eval/prompted_precision'] = precision.cpu()
        metrics['eval/prompted_recall'] = recall.cpu()
        metrics['eval/prompted_f1'] = f1.cpu()
      else:
        import ipdb; ipdb.set_trace()
        sample['lcd'] = sample['lcd'][:,:,0]
        paz = arbiter.forward(utils.flat_batch(sample))
        taz = arbiter.forward(utils.flat_batch(batch))
        cosdist = 1 - self.cossim(paz, taz).mean().cpu()
        metrics['eval/cosdist'] = cosdist

  def _lcd_video(self, epoch, writer, pred, truth=None, name=None):
    """visualize lcd reconstructions"""
    # visualization
    pred_lcds = pred[:8].cpu().detach().numpy()
    if truth is not None:
      real_lcds = truth[:8].cpu().detach().numpy()
      error = (pred_lcds - real_lcds + 1.0) / 2.0
      blank = np.zeros_like(real_lcds)[..., :1, :]
      out = np.concatenate([real_lcds, blank, pred_lcds, blank, error], 3)
      name = name or 'prompted_lcd'
    else:
      out = pred_lcds
      name = name or 'unprompted_lcd'
    out = out.repeat(4, -1).repeat(4, -2)
    utils.add_video(writer, name, utils.force_shape(out), epoch, fps=self.G.fps)

  def _proprio_video(self, epoch, writer, pred, truth=None, name=None):
    """visualize proprio reconstructions"""
    pred_proprio = pred[:8].cpu().detach().numpy()
    pred_lcds = []
    for i in range(pred_proprio.shape[1]):
      lcd = self.venv.reset(np.arange(8), proprio=pred_proprio[:, i])['lcd']
      pred_lcds += [lcd]
    pred_lcds = 1.0 * np.stack(pred_lcds, 1)[:, :, None]

    if truth is not None:
      true_proprio = truth[:8].cpu().detach().numpy()
      true_lcds = []
      for i in range(true_proprio.shape[1]):
        lcd = self.venv.reset(np.arange(8), proprio=true_proprio[:, i])['lcd']
        true_lcds += [lcd]
      true_lcds = 1.0 * np.stack(true_lcds, 1)[:, :, None]
      error = (pred_lcds - true_lcds + 1.0) / 2.0
      blank = np.zeros_like(true_lcds)[..., :1, :]
      out = np.concatenate([true_lcds, blank, pred_lcds, blank, error], 3)
      name = name or 'prompted_proprio'
    else:
      name = name or 'unprompted_proprio'
      out = pred_lcds
    out = out.repeat(4, -1).repeat(4, -2)
    utils.add_video(writer, name, utils.force_shape(out), epoch, fps=self.G.fps)
