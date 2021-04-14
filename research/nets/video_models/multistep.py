import contextlib
from os import EX_TEMPFAIL
import sys
from collections import defaultdict
from typing import IO
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import torch as torchvision
from torch.optim import Adam
from itertools import chain, count
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead, aggregate, MultiHead, ConvEmbed
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .multi_enc import MultiEnc
from .gpt import GPT
import utils

class Multistep(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    self.G = G
    self.act_n = env.action_space.shape[0]
    self.multi_encoder = MultiEnc(env, G)
    self.optimizer = Adam(self.multi_encoder.parameters(), lr=G.lr)
    self.num_stack_tokens = G.imsize // 16  # encoder downsizes 4x over height,width
    self.block_size = self.num_stack_tokens * G.stacks_per_block
    self.gpt = GPT(G.vqK, self.block_size, cond_size=self.G.n_embed, G=G)
    self.act_preproc = nn.Sequential(
      nn.Linear(self.act_n, self.G.hidden_size),
      nn.ReLU(),
      nn.Linear(self.G.hidden_size, self.G.n_embed, bias=False),
    )
    self.gpt_optimizer = Adam(chain(self.act_preproc.parameters(), self.gpt.parameters()), lr=G.lr)  # , betas=(0.5, 0.999))
    self.scaler = th.cuda.amp.GradScaler(enabled=G.amp)
    self.env = env
    if self.G.phase == 2:
      weight_path = self.G.weightdir
      self.load(weight_path)


    def flatbatch(batch):
      ebatch = {}
      for key, val in batch.items():
        shape = val.shape
        ebatch[key] = val.reshape([shape[0] * self.G.stacks_per_block, self.G.vidstack, *shape[2:]])
      return ebatch
    self.flatbatch = flatbatch

  def save(self, dir):
    print("SAVED MODEL", dir)
    path = dir / 'multienc.pt'
    th.save(self.multi_encoder.state_dict(), path)
    print(path)

  def load(self, path):
    enc_path = path / 'multienc.pt'
    self.multi_encoder.load_state_dict(th.load(enc_path))
    print(f'LOADED {enc_path}')

  def forward(self, batch):
    import ipdb; ipdb.set_trace()

  def train_step(self, batch, dry=False):
    """dry means don't update"""
    bs = batch['pstate'].shape[0]
    if self.G.phase == 1:
      #import ipdb; ipdb.set_trace()
      #ebatch = self.flatbatch(batch)
      self.optimizer.zero_grad()
      loss, metrics = self.multi_encoder.loss(batch, return_idxs=True)
      idxs = metrics.pop('idxs').detach().flatten(-2)
      if not dry:
        loss.backward()
        self.optimizer.step()
    elif self.G.phase == 2:
      metrics = {}
      self.gpt_optimizer.zero_grad()
      if self.G.amp:
        context = th.cuda.amp.autocast()
      else:
        context = contextlib.suppress()
      with context:
        with th.no_grad():
          ebatch = self.flatbatch(batch)
          _, decoded, _, idxs = self.multi_encoder.forward(ebatch)
        # PRIOR
        code_idxs = F.one_hot(idxs, self.G.vqK).float()
        code_idxs = code_idxs.reshape([bs, self.block_size, self.G.vqK])
        acts = self.act_preproc(batch['acts'])
        acts = acts.reshape([bs, -1, self.G.vidstack, self.G.n_embed]).mean(2)
        acts = acts.repeat_interleave(self.num_stack_tokens, 1)
        gpt_dist = self.gpt.forward(code_idxs, cond=acts)
        prior_loss = -gpt_dist.log_prob(code_idxs).mean()
        if not dry:
          self.optimizer.step()
          self.scaler.scale(prior_loss).backward()
          #self.scaler.unscale_(self.gpt_optimizer)
          #clip grads if want
          self.scaler.step(self.gpt_optimizer)
          self.scaler.update()
      #prior_loss.backward()
      #self.gpt_optimizer.step()
      metrics['prior_loss'] = prior_loss
    return metrics

  def evaluate(self, writer, batch, epoch):
    bs = batch['pstate'].shape[0]
    #ebatch = self.flatbatch(batch)
    if self.G.phase == 1:
      ebatch = batch
    else:
      ebatch = self.flatbatch(batch)
    _, decoded, _, idxs = self.multi_encoder.forward(ebatch)
    pred_lcd = 1.0 * (decoded['lcd'].probs > 0.5)[:8]
    lcd = ebatch['lcd'][:8]
    error = (pred_lcd - lcd + 1.0) / 2.0
    stack = th.cat([lcd, pred_lcd, error], -2)[0][:, None]
    writer.add_image('image/decode', utils.combine_imgs(stack, 1, self.G.vidstack)[None], epoch)

    pred_state = decoded['pstate'].mean[0].detach().cpu()
    true_state = ebatch['pstate'][0].cpu()
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
    writer.add_image('pstate/decode', utils.combine_imgs(stack, 1, self.G.vidstack)[None], epoch)

    if self.G.phase == 2:
      idxs = idxs.detach().flatten(-2)
      code_idxs = F.one_hot(idxs, self.G.vqK).float()
      code_idxs = code_idxs.reshape([bs, self.block_size, self.G.vqK])[:5]
      for i in range(2 * self.num_stack_tokens, self.G.stacks_per_block * self.num_stack_tokens):
        code_idxs[:, i] = self.gpt.forward(code_idxs).sample()[:, i]

      prior_enc = self.multi_encoder.vq.idx_to_encoding(code_idxs).reshape([5 * self.G.stacks_per_block, 4, 6, self.G.vqD])
      decoded = self.multi_encoder.decoder(prior_enc.permute(0, 3, 1, 2))
      sample_lcd = 1.0 * (decoded['lcd'].probs > 0.5)
      sample_lcd = sample_lcd.reshape([5, self.G.stacks_per_block * self.G.vidstack, self.G.lcd_h, self.G.lcd_w])
      lcd = batch['lcd'][:5]
      error = (sample_lcd - lcd + 1.0) / 2.0
      stack = th.cat([lcd, sample_lcd, error], -2)
      writer.add_video('video_sample', utils.combine_imgs(stack[:, :, None], 1, 5)[None, :, None], epoch, fps=60)
