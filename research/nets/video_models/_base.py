from re import I
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
from research.nets._base import Net
from research import utils
from research.wrappers.async_vector_env import AsyncVectorEnv
from research.define_config import env_fn
from jax.tree_util import tree_map
A = utils.A
RED, GREEN = A[0.9, 0.2, 0.2], A[0.2, 0.9, 0.2]

class VideoModel(Net):
  def __init__(self, env, G):
    super().__init__(G)
    self.env = env
    self.act_n = env.action_space.shape[0]
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    self.proprio_n = env.observation_space.spaces['proprio'].shape[0]
    self.venv = AsyncVectorEnv([env_fn(G) for _ in range(self.G.video_n)])

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
      t_post_prompt = tree_map(lambda x: x[:, self.G.prompt_n:], batch)
      s_post_prompt = tree_map(lambda x: x[:, self.G.prompt_n:], sample)
      s_post_prompt['lcd'] = s_post_prompt['lcd'][:, :, 0]
      if arbiter.original_name == 'TracedArbiter':
        def chop(x):
          T = x.shape[1]
          _chop = T % arbiter.G.window
          if _chop != 0:
            x = x[:, :-_chop]
          return x.reshape([-1, arbiter.G.window, *x.shape[2:]])
        s_window = tree_map(chop, s_post_prompt)
        t_window = tree_map(chop, t_post_prompt)
        sact = action[:, self.G.prompt_n:]
        tact = batch['action'][:, self.G.prompt_n:]
        sact = chop(sact)[:, :-1]
        tact = chop(tact)[:, :-1]

        paz, paa = arbiter.forward(s_window)
        metrics['eval/unprompted_action_log_mse'] = ((sact - paa)**2).mean().log()
        taz, taa = arbiter.forward(t_window)
        fvd = utils.compute_fid(paz.cpu().numpy(), taz.cpu().numpy())
        metrics['eval/unprompted_fvd'] = fvd
        precision, recall, f1 = utils.precision_recall_f1(taz, paz, k=5)
        metrics['eval/unprompted_precision'] = precision.cpu()
        metrics['eval/unprompted_recall'] = recall.cpu()
        metrics['eval/unprompted_f1'] = f1.cpu()
      else:
        paz = arbiter.forward(utils.flat_batch(s_post_prompt)).cpu().numpy()
        taz = arbiter.forward(utils.flat_batch(t_post_prompt)).cpu().numpy()
        fid = utils.compute_fid(paz, taz)
        metrics['eval/fid'] = fid

  def _duplicate_eval(self, epoch, writer, metrics, batch, arbiter=None):
    n = batch['lcd'].shape[0]
    batch = {key: val[:1].repeat_interleave(self.G.video_n, 0) for key, val in batch.items()}
    sample = self.sample(n, action=batch['action'], prompts=batch, prompt_n=self.G.prompt_n)
    if 'lcd' in sample:
      pred_lcd = sample['lcd']
      true_lcd = batch['lcd'][:, :, None]
      self._lcd_video(epoch, writer, pred_lcd, true_lcd, name='duplicate_lcd', prompt_n=self.G.prompt_n)
    if 'proprio' in sample:
      pred_proprio = sample['proprio']
      true_proprio = batch['proprio']
      self._proprio_video(epoch, writer, pred_proprio, true_proprio, name='duplicate_proprio', prompt_n=self.G.prompt_n)

  def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
    n = batch['lcd'].shape[0]
    sample = self.sample(n, action=batch['action'], prompts=batch, prompt_n=self.G.prompt_n)

    if 'lcd' in sample:
      pred_lcd = sample['lcd'][:, self.G.prompt_n:]
      true_lcd = batch['lcd'][:, :, None][:, self.G.prompt_n:]
      # run basic metrics
      self.ssim.update((pred_lcd.flatten(0, 1), true_lcd.flatten(0, 1)))
      ssim = self.ssim.compute().cpu().detach()
      metrics['eval/ssim'] = ssim
      # TODO: try not flat
      self.psnr.update((pred_lcd.flatten(0, 1), true_lcd.flatten(0, 1)))
      psnr = self.psnr.compute().cpu().detach()
      metrics['eval/psnr'] = psnr
      # visualize reconstruction
      self._lcd_video(epoch, writer, pred_lcd, true_lcd, prompt_n=self.G.prompt_n)

    if 'proprio' in sample:
      pred_proprio = sample['proprio']
      true_proprio = batch['proprio']
      metrics['eval/proprio_log_mse'] = ((true_proprio[:, self.G.prompt_n:] - pred_proprio[:, self.G.prompt_n:])**2).mean().log().cpu()
      # visualize reconstruction
      self._proprio_video(epoch, writer, pred_proprio, true_proprio, prompt_n=self.G.prompt_n)

    if arbiter is not None:
      t_post_prompt = tree_map(lambda x: x[:, self.G.prompt_n:], batch)
      s_post_prompt = tree_map(lambda x: x[:, self.G.prompt_n:], sample)
      s_post_prompt['lcd'] = s_post_prompt['lcd'][:, :, 0]
      if arbiter.original_name == 'TracedArbiter':

        def chop(x):
          T = x.shape[1]
          _chop = T % arbiter.G.window
          if _chop != 0:
            x = x[:, :-_chop]
          return x.reshape([-1, arbiter.G.window, *x.shape[2:]])
        s_window = tree_map(chop, s_post_prompt)
        t_window = tree_map(chop, t_post_prompt)
        tact = batch['action'][:, self.G.prompt_n:]
        tact = chop(tact)[:, :-1]

        paz, paa = arbiter.forward(s_window)
        metrics['eval/prompted_action_log_mse'] = ((tact - paa)**2).mean().log()
        taz, taa = arbiter.forward(t_window)
        metrics['eval/true_action_log_mse'] = ((tact - taa)**2).mean().log()
        fvd = utils.compute_fid(paz.cpu().numpy(), taz.cpu().numpy())
        metrics['eval/prompted_fvd'] = fvd
        precision, recall, f1 = utils.precision_recall_f1(taz, paz, k=5)
        metrics['eval/prompted_precision'] = precision.cpu()
        metrics['eval/prompted_recall'] = recall.cpu()
        metrics['eval/prompted_f1'] = f1.cpu()
        cosdist = 1 - self.cossim(paz, taz).mean().cpu()
        metrics['eval/prompted_cosdist'] = cosdist
      else:
        import ipdb; ipdb.set_trace()
        sample['lcd'] = sample['lcd'][:, :, 0]
        paz = arbiter.forward(utils.flat_batch(sample))
        taz = arbiter.forward(utils.flat_batch(batch))
        cosdist = 1 - self.cossim(paz, taz).mean().cpu()
        metrics['eval/cosdist'] = cosdist

  def _lcd_video(self, epoch, writer, pred, truth=None, name=None, prompt_n=None):
    """visualize lcd reconstructions"""
    # visualization
    pred_lcds = pred[:self.G.video_n].cpu().detach().numpy()
    if truth is not None:
      real_lcds = truth[:self.G.video_n].cpu().detach().numpy()
      error = (pred_lcds - real_lcds + 1.0) / 2.0
      blank = np.zeros_like(real_lcds)[..., :1, :]
      out = np.concatenate([real_lcds, blank, pred_lcds, blank, error], 3)
      name = name or 'prompted_lcd'
    else:
      out = pred_lcds
      name = name or 'unprompted_lcd'
    out = utils.force_shape(out)
    out = out.repeat(3, 2)
    if prompt_n is not None:
      W, H = self.G.lcd_w, self.G.lcd_h
      out = out.transpose(0, 1, 4, 3, 2)
      for j in range(7):
        out[:, :prompt_n, W * (j + 1) + j, :] = GREEN
        out[:, prompt_n:, W * (j + 1) + j, :] = RED
      out[:, :prompt_n, :, H] = GREEN
      out[:, prompt_n:, :, H] = RED
      out[:, :prompt_n, :, 2 * H + 1] = GREEN
      out[:, prompt_n:, :, 2 * H + 1] = RED
      out = out.transpose(0, 1, 4, 3, 2)

    #  out[:,:prompt_n] = np.where(out==[0.0,0.0,0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0])[:,:prompt_n]
    out = out.repeat(4, -1).repeat(4, -2)
    utils.add_video(writer, name, out, epoch, fps=self.G.fps)

  def _proprio_video(self, epoch, writer, pred, truth=None, name=None, prompt_n=None):
    """visualize proprio reconstructions"""
    pred_proprio = pred[:self.G.video_n].cpu().detach().numpy()
    pred_lcds = []
    for i in range(pred_proprio.shape[1]):
      lcd = self.venv.reset(np.arange(self.G.video_n), proprio=pred_proprio[:, i])['lcd']
      pred_lcds += [lcd]
    pred_lcds = 1.0 * np.stack(pred_lcds, 1)[:, :, None]

    if truth is not None:
      true_proprio = truth[:self.G.video_n].cpu().detach().numpy()
      true_lcds = []
      for i in range(true_proprio.shape[1]):
        lcd = self.venv.reset(np.arange(self.G.video_n), proprio=true_proprio[:, i])['lcd']
        true_lcds += [lcd]
      true_lcds = 1.0 * np.stack(true_lcds, 1)[:, :, None]
      error = (pred_lcds - true_lcds + 1.0) / 2.0
      blank = np.zeros_like(true_lcds)[..., :1, :]
      out = np.concatenate([true_lcds, blank, pred_lcds, blank, error], 3)
      name = name or 'prompted_proprio'
    else:
      name = name or 'unprompted_proprio'
      out = pred_lcds
    out = utils.force_shape(out)
    out = out.repeat(3, 2)
    if prompt_n is not None:
      W, H = self.G.lcd_w, self.G.lcd_h
      out = out.transpose(0, 1, 4, 3, 2)
      for j in range(7):
        out[:, :prompt_n, W * (j + 1) + j, :] = GREEN
        out[:, prompt_n:, W * (j + 1) + j, :] = RED
      out[:, :prompt_n, :, H] = GREEN
      out[:, prompt_n:, :, H] = RED
      out[:, :prompt_n, :, 2 * H + 1] = GREEN
      out[:, prompt_n:, :, 2 * H + 1] = RED
      out = out.transpose(0, 1, 4, 3, 2)
      #out[:,:prompt_n] = np.where(out==[0.0,0.0,0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0])[:,:prompt_n]
    out = out.repeat(4, -1).repeat(4, -2)
    utils.add_video(writer, name, out, epoch, fps=self.G.fps)
