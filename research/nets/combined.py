import sys
from collections import defaultdict
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
from nets.common import GaussHead, MDNHead, CausalSelfAttention, TransformerBlock, BinaryHead, aggregate, MultiHead, ConvEmbed
import torch as th
from torch import distributions as thd
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vqvae import VQVAE
from .state_vqvae import State_VQVAE
from .gpt import GPT
import utils

class Combined(nn.Module):
  def __init__(self, env, G):
    super().__init__()
    self.vqvae = VQVAE(env, G)
    self.state_vqvae = State_VQVAE(env, G)
    self.gpt = GPT(G.vqK, 8 + 24, G=G)
    self.image_optimizer = Adam(self.vqvae.parameters(), lr=G.lr)
    self.state_optimizer = Adam(self.state_vqvae.parameters(), lr=G.lr)
    self.gpt_optimizer = Adam(self.gpt.parameters(), lr=1e-3)#, betas=(0.5, 0.999))
    self.env = env
    self.G = G

  def forward(self, batch):
    import ipdb; ipdb.set_trace()

  def train_step(self, batch):
    # image
    self.image_optimizer.zero_grad()
    iloss, imetrics = self.vqvae.loss(batch, return_idxs=True)
    iloss.backward()
    self.image_optimizer.step()
    # state
    self.state_optimizer.zero_grad()
    sloss, smetrics = self.state_vqvae.loss(batch, return_idxs=True)
    sloss.backward()
    self.state_optimizer.step()

    # combined prior
    self.gpt_optimizer.zero_grad()
    state_idxs = smetrics.pop('idxs')
    image_idxs = imetrics.pop('idxs')
    idxs = th.cat([state_idxs, image_idxs.flatten(-2)], -1)
    code_idxs = F.one_hot(idxs.detach(), self.G.vqK).float()
    import ipdb; ipdb.set_trace()
    gpt_dist = self.gpt.forward(code_idxs)
    prior_loss = -gpt_dist.log_prob(code_idxs).mean()
    prior_loss.backward()
    self.gpt_optimizer.step()
    # TODO: what about training end2end more?

    return {
        'prior_loss': prior_loss,
        **{'image/' + key: val for key, val in imetrics.items()},
        **{'pstate/' + key: val for key, val in smetrics.items()}
    }

  def evaluate(self, writer, batch, epoch):
    iloss, imetrics = self.vqvae.loss(batch, eval=True, return_idxs=True)
    pred_lcd = 1.0 * (imetrics.pop('decoded')[:8].exp() > 0.5)
    lcd = batch['lcd'][:8]
    error = (pred_lcd - lcd + 1.0) / 2.0
    stack = th.cat([lcd, pred_lcd, error], -2)
    writer.add_image('image/decode', utils.combine_imgs(stack, 1, 8)[None], epoch)

    sloss, smetrics = self.state_vqvae.loss(batch, eval=True, return_idxs=True)
    pred_state = smetrics.pop('decoded').mean[:8].detach().cpu()
    true_state = batch['pstate'][:8].cpu()
    preds = []
    for s in pred_state:
      preds += [self.env.reset(state=s)['lcd']]
    truths = []
    for s in true_state:
      truths += [self.env.reset(state=s)['lcd']]
    preds = 1.0 * np.stack(preds)
    truths = 1.0 * np.stack(truths)
    error = (preds - truths + 1.0) / 2.0
    stack = np.concatenate([truths, preds, error], -2)[:, None]
    writer.add_image('pstate/decode', utils.combine_imgs(stack, 1, 8)[None], epoch)

    # SAMPLE
    # image based on state
    # TODO: test with original idxs. no changes.
    image_idxs = imetrics.pop('idxs')
    state_idxs = smetrics.pop('idxs')
    state_idxs[4:] = state_idxs[4:5,:] # make the last 4 be all the same
    idxs = th.cat([state_idxs, th.zeros_like(image_idxs.flatten(-2))], -1)
    code_idxs = F.one_hot(idxs.detach(), self.G.vqK).float()
    for i in range(8, self.gpt.block_size):
      dist = self.gpt.forward(code_idxs)
      code_idxs[:, i] = dist.sample()[:, i]
    sample_image_idxs = code_idxs[:, 8:]
    prior_enc = self.vqvae.vq.idx_to_encoding(sample_image_idxs).reshape([-1, 4, 6, self.G.vqD]).permute(0, 3, 1, 2)
    #prior_enc = self.vqvae.vq.idx_to_encoding(sample_image_idxs).permute(0,2,1).reshape([-1, self.G.vqD, 4, 6])
    decoded = self.vqvae.decoder(prior_enc)[:8]
    sample_lcd = 1.0 * (decoded.exp() > 0.5)
    lcd[4:] = lcd[4:5] # make the last 4 be all the same
    error = (sample_lcd - lcd + 1.0) / 2.0
    stack = th.cat([lcd, sample_lcd, error], -2)
    writer.add_image('pstate2image_sample', utils.combine_imgs(stack, 1, 8)[None], epoch)
