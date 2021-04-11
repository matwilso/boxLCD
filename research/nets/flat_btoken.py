from re import I
import yaml
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from nets.common import Block, BinaryHead
import utils
from nets.bvae import BVAE

class FlatBToken(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.env = env
    self.act_n = env.action_space.shape[0]
    self.observation_space = env.observation_space
    self.action_space = env.action_space
    self.pstate_n = env.observation_space.spaces['pstate'].shape[0]
    self.block_size = self.C.window

    # LOAD SVAE
    sd = th.load(C.weightdir/'bvae.pt')
    svaeC = sd.pop('C')
    self.bvae = BVAE(env, svaeC)
    self.bvae.load(C.weightdir)
    for p in self.bvae.parameters():
      p.requires_grad = False
    self.bvae.eval()
    # </LOAD SVAE>

    self.size = self.bvae.C.vqD * 4 * 8
    self.gpt_size = self.size
    self.dist = self.C.decode
    self.block_size = self.C.window

    self.pos_emb = nn.Parameter(th.zeros(1, self.block_size, C.n_embed))
    self.cond_in = nn.Linear(self.act_n, C.n_embed // 2, bias=False)
    # input embedding stem
    self.embed = nn.Linear(self.gpt_size, C.n_embed // 2, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[Block(self.block_size, C) for _ in range(C.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(C.n_embed)
    self.dist_head = BinaryHead(C.n_embed, self.gpt_size, C)
    self.optimizer = Adam(self.parameters(), lr=C.lr)
    self.to(C.device)

  def save(self, dir):
    print("SAVED MODEL", dir)
    path = dir / 'bvae.pt'
    sd = self.state_dict()
    sd['C'] = self.C
    th.save(sd, path)
    print(path)
    self.bvae.save(dir)

  def load(self, dir):
    path = dir / 'bvae.pt'
    sd = th.load(path)
    C = sd.pop('C')
    self.load_state_dict(sd)
    self.bvae.load(dir)
    print(f'LOADED {path}')

  def forward(self, batch):
    BS, EPL, *HW = batch['lcd'].shape
    acts = batch['acts']
    flatter_batch = {key: val.flatten(0, 1) for key, val in batch.items()}
    z_q = self.bvae.encode(flatter_batch)
    z_q = z_q.reshape([BS, EPL, -1])
    x = self.embed(z_q)
    # forward the GPT model
    BS, T, E = x.shape
    # SHIFT RIGHT (add a padding on the left)
    x = th.cat([th.zeros(BS, 1, E).to(self.C.device), x[:, :-1]], dim=1)
    acts = th.cat([th.zeros(BS, 1, acts.shape[-1]).to(self.C.device), acts[:,:-1]], dim=1)
    cin = self.cond_in(acts)
    if acts.ndim == 2:
      x = th.cat([x, cin[:, None].repeat_interleave(self.block_size, 1)], -1)
    else:
      x = th.cat([x, cin], -1)
    x += self.pos_emb  # each position maps to a (learnable) vector
    x = self.blocks(x)
    logits = self.ln_f(x)
    return logits

  def train_step(self, batch, dry=False):
    self.optimizer.zero_grad()
    loss, metrics = self.loss(batch)
    if not dry:
      loss.backward()
      self.optimizer.step()
    return metrics

  def loss(self, batch):
    BS, EPL, *HW = batch['lcd'].shape
    metrics = {}
    logits = self.forward(batch)
    dist = self.dist_head(logits)
    z_q = self.bvae.encode({key: val.flatten(1, 1) for key, val in batch.items()})
    z_q = z_q.reshape([BS, EPL, -1]).detach()
    loss = -dist.log_prob(z_q).mean()
    metrics['loss/total'] = loss
    return loss, metrics

  def onestep(self, batch, i, temp=1.0):
    logits = self.forward(batch)
    dist = self.dist_head(logits / temp)
    import ipdb; ipdb.set_trace()
    sample = dist.sample()
    batch['lcd'][:, i] = sample[:, i, :self.C.imsize].reshape(batch['lcd'][:,i].shape)
    pstate_code = sample[:, i, self.C.imsize:]
    pstate = self.bvae.decoder(pstate_code).mean
    batch['pstate'][:, i] = pstate
    return batch

  def sample(self, n, acts=None, prompts=None, prompt_n=10):
    # TODO: feed act_n
    with th.no_grad():
      if acts is not None:
        n = acts.shape[0]
      batch = {}
      batch['lcd'] = th.zeros(n, self.block_size, self.C.lcd_h, self.C.lcd_w).to(self.C.device)
      batch['pstate'] = th.zeros(n, self.block_size, self.pstate_n).to(self.C.device)
      batch['acts'] = acts if acts is not None else (th.rand(n, self.block_size, self.act_n) * 2 - 1).to(self.C.device)
      start = 0
      if prompts is not None:
        batch['lcd'][:, :10] = prompts['lcd'][:, :prompt_n]
        batch['pstate'][:, :10] = prompts['pstate'][:, :prompt_n]
        start = prompt_n

      for i in range(start, self.block_size):
        # TODO: check this setting since we have modified things
        logits = self.forward(batch)
        dist = self.dist_head(logits)
        sample = dist.sample()
        sample_ze = sample.reshape([n*self.block_size, self.bvae.C.vqD, 4, 8])
        dist = self.bvae.decoder(sample_ze)
        # grab the modes since we don't want to sample in the end space, just latent
        sample_lcd = (1.0*(dist['lcd'].probs > 0.5)).reshape([n, self.block_size, self.C.lcd_h, self.C.lcd_w])
        sample_pstate = (dist['pstate'].mean).reshape([n, self.block_size, -1])
        batch['lcd'][:, i] = sample_lcd[:, i]
        batch['pstate'][:, i] = sample_pstate[:, i]
        if i == self.block_size - 1:
          sample_loss = self.loss(batch)[0]
    batch['lcd'] = batch['lcd'].reshape(n, -1, 1, self.C.lcd_h, self.C.lcd_w)
    return batch, sample_loss.mean().cpu().detach()

  def evaluate(self, writer, batch, epoch):
    N = self.C.num_envs
    # unpropted
    acts = (th.rand(N, self.C.window, self.env.action_space.shape[0]) * 2 - 1).to(self.C.device)
    sample, sample_loss = self.sample(N, acts=acts)
    lcd = sample['lcd']
    lcd = lcd.cpu().detach().repeat_interleave(4, -1).repeat_interleave(4, -2)[:, 1:]
    #writer.add_video('lcd_samples', utils.force_shape(lcd), epoch, fps=self.C.fps)
    utils.add_video(writer, 'lcd_samples', utils.force_shape(lcd), epoch, fps=self.C.fps)
    # prompted
    if len(self.env.world_def.robots) == 0:  # if we are just dropping the object, always use the same setup
      if 'BoxOrCircle' == self.C.env:
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
      for _ in range(self.C.window - 1):
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
    acts = th.as_tensor(acts, dtype=th.float32).to(self.C.device)
    prompts = {key: th.as_tensor(1.0 * val[:, :10]).to(self.C.device) for key, val in obses.items()}
    # dupe
    # for key in prompts:
    #  prompts[key][4:] = prompts[key][4:5]
    acts[4:] = acts[4:5]
    prompted_samples, prompt_loss = self.sample(N, acts=acts, prompts=prompts)
    real_lcd = obses['lcd'][:, :, None]
    lcd_psamp = prompted_samples['lcd']
    lcd_psamp = lcd_psamp.cpu().detach().numpy()
    error = (lcd_psamp - real_lcd + 1.0) / 2.0
    blank = np.zeros_like(real_lcd)[..., :1, :]
    out = np.concatenate([real_lcd, blank, lcd_psamp, blank, error], 3)
    out = out.repeat(4, -1).repeat(4, -2)
    #writer.add_video('prompted_lcd', utils.force_shape(out), epoch, fps=self.C.fps)
    utils.add_video(writer, 'prompted_lcd', utils.force_shape(out), epoch, fps=self.C.fps)

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
    utils.add_video(writer, 'prompted_state', utils.force_shape(out), epoch, fps=self.C.fps)
