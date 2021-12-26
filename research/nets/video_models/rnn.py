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

# RNN

H = 128
class SingleState(VideoModel):
  def __init__(self, env, G):
    super().__init__(env, G)
    size = env.observation_space['full_state'].shape[0]
    self.lstm = nn.LSTM(size, H, num_layers=1, batch_first=True)
    self.fc = nn.Linear(H, size)
    self.optimizer = Adam(self.parameters(), lr=G.lr)

  def forward(self, batch):
    import ipdb; ipdb.set_trace()
    # Forward propagate LSTM
    out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (bs, seq_length, hidden_size)

    # Decode the hidden state of the last time step
    out = self.fc(out).squeeze(-1)  # b x 784
    loss = -tdib.Bernoulli(logits=out.reshape([bs, 1, 28, 28])).log_prob(inp).mean()
    return loss, {'nlogp': loss}
    import ipdb; ipdb.set_trace()

  def train_step(self, batch, dry=False):
    self.optimizer.zero_grad()
    loss, metrics = self.loss(batch)
    if not dry:
      loss.backward()
      self.optimizer.step()
    return metrics

  def loss(self, batch):
    state = batch['full_state'] # BS x WINDOW x SIZE
    # zero pad window
    state = th.cat([th.zeros(state.shape[0], 1, state.shape[2]).to(state.device), state], dim=1)
    out, _ = self.lstm(state)
    out = self.fc(out)
    mse = (state[:,1:] - out[:,:-1]).pow(2).mean()
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
        out, (h, c) = self.lstm(th.zeros(n, 1, 4).to(self.device))
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



