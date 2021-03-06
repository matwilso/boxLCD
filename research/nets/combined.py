import sys
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch as th
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead, aggregate, MultiHead, ConvEmbed
import torch as th
from torch import distributions as tdib
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vqvae import VQVAE
from .state_vqvae import State_VQVAE
import utils

class Combined(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.vqvae = VQVAE(env, C)
    self.state_vqvae = State_VQVAE(env, C)

    self.image_optimizer = Adam(self.vqvae.parameters(), lr=C.lr)
    self.state_optimizer = Adam(self.state_vqvae.parameters(), lr=C.lr)
    self.env = env

  def forward(self, x):
    import ipdb; ipdb.set_trace()

  def train_step(self, batch):
    self.image_optimizer.zero_grad()
    iloss, imetrics = self.vqvae.loss(batch)
    iloss.backward()
    self.image_optimizer.step()

    self.state_optimizer.zero_grad()
    sloss, smetrics = self.state_vqvae.loss(batch)
    sloss.backward()
    self.state_optimizer.step()
    return {
      **{'image/'+key: val for key, val in imetrics.items()},
      **{'state/'+key: val for key, val in smetrics.items()}
      }

  def evaluate(self, writer, batch, epoch):
    iloss, imetrics = self.vqvae.loss(batch, eval=True)
    pred_lcd = 1.0*(imetrics.pop('decoded')[:8].exp() > 0.5)
    lcd = batch['lcd'][:8]
    error = (pred_lcd - lcd + 1.0) / 2.0
    stack = th.cat([lcd, pred_lcd, error], -2)
    writer.add_image('image/decode', utils.combine_imgs(stack, 1, 8)[None], epoch)

    sloss, smetrics = self.state_vqvae.loss(batch, eval=True)
    pred_state = smetrics.pop('decoded').mean[:8].detach().cpu()
    true_state = batch['state'][:8].cpu()
    preds = []
    for s in pred_state:
      preds += [self.env.reset(state=s)['lcd']]
    truths = []
    for s in true_state:
      truths += [self.env.reset(state=s)['lcd']]
    preds = 1.0*np.stack(preds)
    truths = 1.0*np.stack(truths)
    error = (preds - truths + 1.0) / 2.0
    stack = np.concatenate([truths, preds, error], -2)[:,None]
    writer.add_image('state/decode', utils.combine_imgs(stack, 1, 8)[None], epoch)
