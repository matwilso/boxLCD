import numpy as np
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
from research.nets._base import Net
from research import utils
import ignite

class VideoModel(Net):
  def __init__(self, env, G):
    super().__init__(G)
    self.env = env
    self.act_n = env.action_space.shape[0]
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    self.pstate_n = env.observation_space.spaces['pstate'].shape[0]
    self.ssim = ignite.metrics.SSIM(1.0, device=self.G.device)
    self.psnr = ignite.metrics.PSNR(1.0, device=self.G.device)

  def onestep(self):
    raise NotImplementedError

  def sample(self, n, acts=None, prompts=None, prompt_n=10):
    raise NotImplementedError

  def evaluate(self, batch, itr):
    N = self.G.num_envs
    # UNPROMPTED SAMPLE
    acts = (th.rand(N, self.G.window, self.env.action_space.shape[0]) * 2 - 1).to(self.G.device)
    sample, sample_loss = self.model.sample(N, acts=acts)
    if 'lcd' in sample:
      lcd = sample['lcd']
      lcd = lcd.cpu().detach().repeat_interleave(4, -1).repeat_interleave(4, -2)[:, 1:]
      #writer.add_video('lcd_samples', utils.force_shape(lcd), itr, fps=self.G.fps)
      utils.add_video(self.writer, 'unprompted/lcd', utils.force_shape(lcd), itr, fps=self.G.fps)
    if 'pstate' in sample:
      # TODO: add support for pstate
      import ipdb; ipdb.set_trace()

    # PROMPTED SAMPLE
    prompted_samples, prompt_loss = self.model.sample(N, acts=batch['acts'], prompts=batch, prompt_n=10)
    pred_lcd = prompted_samples['lcd']
    real_lcd = batch['lcd'][:,:,None]
    # metrics 
    self.ssim.update((pred_lcd.flatten(0,1), real_lcd.flatten(0,1)))
    ssim = self.ssim.compute().cpu().detach()
    self.logger['eval/ssim'] += [ssim]
    # TODO: try not flat
    self.psnr.update((pred_lcd.flatten(0,1), real_lcd.flatten(0,1)))
    psnr = self.psnr.compute().cpu().detach()
    self.logger['eval/psnr'] += [psnr]

    if 'pstate' in prompted_samples:
      import ipdb; ipdb.set_trace()
      pstate_samp = prompted_samples['pstate'].cpu().numpy()
      imgs = []
      shape = pstate_samp.shape
      for ii in range(shape[0]):
        col = []
        for jj in range(shape[1]):
          col += [self.env.reset(pstate=pstate_samp[ii, jj])['lcd']]
        imgs += [col]
      pstate_img = 1.0 * np.array(imgs)[:, :, None]
      error = (pstate_img - real_lcd + 1.0) / 2.0
      blank = np.zeros_like(real_lcd)[..., :1, :]
      out = np.concatenate([real_lcd, blank, pstate_img, blank, error], 3)
      out = out.repeat(4, -1).repeat(4, -2)
      utils.add_video(writer, 'prompted_state', utils.force_shape(out), epoch, fps=self.G.fps)


    # visualization
    pred_lcd = pred_lcd[:8].cpu().detach().numpy()
    real_lcd = real_lcd[:8].cpu().detach().numpy()
    error = (pred_lcd - real_lcd + 1.0) / 2.0
    blank = np.zeros_like(real_lcd)[..., :1, :]
    out = np.concatenate([real_lcd, blank, pred_lcd, blank, error], 3)
    out = out.repeat(4, -1).repeat(4, -2)
    #writer.add_video('prompted_lcd', utils.force_shape(out), itr, fps=self.G.fps)
    utils.add_video(self.writer, 'prompted_lcd', utils.force_shape(out), itr, fps=self.G.fps)
    return prompted_samples['lcd']


  def evaluate(self, writer, batch, epoch):
    import ipdb; ipdb.set_trace()
    N = self.G.num_envs
    # unpropted
    acts = (th.rand(N, self.G.window, self.env.action_space.shape[0]) * 2 - 1).to(self.G.device)
    sample = self.sample(N, acts=acts)
    lcd = sample['lcd']
    lcd = lcd.cpu().detach().repeat_interleave(4, -1).repeat_interleave(4, -2)[:, 1:]
    #writer.add_video('lcd_samples', utils.force_shape(lcd), epoch, fps=self.G.fps)
    utils.add_video(writer, 'lcd_samples', utils.force_shape(lcd), epoch, fps=self.G.fps)
    # prompted
    if len(self.env.world_def.robots) == 0:  # if we are just dropping the object, always use the same setup
      if 'BoxOrCircle' == self.G.env:
        reset_states = np.c_[np.ones(N), np.zeros(N), np.linspace(-0.8, 0.8, N), 0.5 * np.ones(N)]
      else:
        reset_states = np.c_[np.random.uniform(-1, 1, N), np.random.uniform(-1, 1, N), np.linspace(-0.8, 0.8, N), 0.5 * np.ones(N)]
    else:
      reset_states = [None] * N
    obses = {key: [[] for ii in range(N)] for key in self.env.observation_space.spaces}
    acts = [[] for ii in range(N)]
    for ii in range(N):
      for key, val in self.env.reset(reset_states[ii]).items():
        obses[key][ii] += [val]
      for _ in range(self.G.window - 1):
        act = self.env.action_space.sample()
        obs = self.env.step(act)[0]
        for key, val in obs.items():
          obses[key][ii] += [val]
        acts[ii] += [act]
      acts[ii] += [np.zeros_like(act)]
    obses = {key: np.array(val) for key, val in obses.items()}
    # dupe
    for key in obses:
      obses[key][4:] = obses[key][4:5]
    acts = np.array(acts)
    acts = th.as_tensor(acts, dtype=th.float32).to(self.G.device)
    prompts = {key: th.as_tensor(1.0 * val[:, :10]).to(self.G.device) for key, val in obses.items()}
    # dupe
    # for key in prompts:
    #  prompts[key][4:] = prompts[key][4:5]
    acts[4:] = acts[4:5]
    prompted_samples = self.sample(N, acts=acts, prompts=prompts)
    real_lcd = obses['lcd'][:, :, None]
    lcd_psamp = prompted_samples['lcd']
    lcd_psamp = lcd_psamp.cpu().detach().numpy()
    error = (lcd_psamp - real_lcd + 1.0) / 2.0
    blank = np.zeros_like(real_lcd)[..., :1, :]
    out = np.concatenate([real_lcd, blank, lcd_psamp, blank, error], 3)
    out = out.repeat(4, -1).repeat(4, -2)
    #writer.add_video('prompted_lcd', utils.force_shape(out), epoch, fps=self.G.fps)
    utils.add_video(writer, 'prompted_lcd', utils.force_shape(out), epoch, fps=self.G.fps)

    pstate_samp = prompted_samples['pstate'].cpu().numpy()
    imgs = []
    shape = pstate_samp.shape
    for ii in range(shape[0]):
      col = []
      for jj in range(shape[1]):
        col += [self.env.reset(pstate=pstate_samp[ii, jj])['lcd']]
      imgs += [col]
    pstate_img = 1.0 * np.array(imgs)[:, :, None]
    error = (pstate_img - real_lcd + 1.0) / 2.0
    blank = np.zeros_like(real_lcd)[..., :1, :]
    out = np.concatenate([real_lcd, blank, pstate_img, blank, error], 3)
    out = out.repeat(4, -1).repeat(4, -2)
    utils.add_video(writer, 'prompted_state', utils.force_shape(out), epoch, fps=self.G.fps)
