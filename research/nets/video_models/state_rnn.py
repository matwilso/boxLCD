from re import I
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
from research.nets.common import GaussHead, MDNHead, CausalSelfAttention, TransformerBlock, BinaryHead, aggregate, MultiHead, ConvEmbed, ConvBinHead
from research import utils
import ignite
from ._base import VideoModel
import boxLCD.utils as boxLCD_utils

# RNN

H = 128
class SingleState(VideoModel):
  def __init__(self, env, G):
    super().__init__(env, G)
    size = env.observation_space['full_state'].shape[0]
    self.lstm = nn.LSTM(size, H, num_layers=1, batch_first=True)
    self.fc = nn.Linear(H, size)
    self.optimizer = Adam(self.parameters(), lr=G.lr)
    self.shape_idxs = [v for k, v in self.env.obs_idx_dict.items() if 'shape' in k]

  def forward(self, batch):
    import ipdb; ipdb.set_trace()
    pass

  def loss(self, batch):
    state = batch['full_state'] # BS x WINDOW x SIZE
    # zero pad window
    state = th.cat([th.zeros(state.shape[0], 1, state.shape[2]).to(state.device), state], dim=1)
    out, _ = self.lstm(state)
    out = self.fc(out)
    delta = state[:,1:] - out[:,:-1]
    #delta[..., self.shape_idxs] = 0.0
    mse = (delta).pow(2).mean()
    return mse, {'mse': mse}

  def onestep(self, batch, i, temp=1.0):
    state = batch['full_state']
    import ipdb; ipdb.set_trace()
    logits = self.forward(batch)
    dist = self.dist_head(logits/temp)
    batch['lcd'][:, i] = dist.sample()[:, i]
    return batch

  def sample(self, n, action=None, prompts=None, prompt_n=10):
    # TODO: feed act_n
    with th.no_grad():
      if action is not None:
        n = action.shape[0]
      batch = {}
      if prompts is None:
        # get initial state by sampling with zeros
        out, (h, c) = self.lstm(th.zeros(n, 1, len(self.env.obs_keys)).to(self.device))
        out = self.fc(out)
        imdts = [out]
        for i in range(self.window-1):
          out, (h, c) = self.lstm(out, (h,c))
          out = self.fc(out)
          imdts += [out]
        batch['full_state'] = th.cat(imdts, 1)
      else:
        out, (h, c) = self.lstm(prompts['full_state'])
        out = self.fc(out)
        batch['full_state'] = out
    return batch