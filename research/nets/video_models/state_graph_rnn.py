from re import I
import sys
from collections import defaultdict
from types import MemberDescriptorType
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
from ..common import TransformerBlock

# State Graph RNN

H = 128
class StateGraphRNN(VideoModel):
  def __init__(self, env, G):
    super().__init__(env, G)
    self.n_objects = len(env.world_def.objects)
    self.size = env.observation_space['full_state'].shape[0] // self.n_objects
    self.embed = nn.Linear(self.size, G.n_embed)
    self.tf = TransformerBlock(self.n_objects, G, causal=False)
    self.lstm = nn.LSTM(G.n_embed, H, num_layers=1, batch_first=True)
    self.fc = nn.Linear(H, self.size)
    self.optimizer = Adam(self.parameters(), lr=G.lr)
    self.shape_idxs = [v for k, v in self.env.obs_idx_dict.items() if 'shape' in k]

  def forward(self, state, hc=None):
    objs = state.unflatten(-1, [self.n_objects, self.size])
    bs, t, n_objects, size = objs.shape
    def flat(x):
      return x.swapaxes(1,2).flatten(0,1)
    def unflat(x):
      return x.unflatten(0, [bs, n_objects]).swapaxes(1,2)
    x = self.embed(objs)
    x = x.flatten(0,1)
    comms = self.tf(x).unflatten(0, [bs,t])
    lstm_in = flat(comms)
    out, (h,c) = self.lstm(lstm_in, hc)
    out = self.fc(out)
    out = unflat(out)
    out = out.flatten(-2,-1)
    return out, (h,c)

  def loss(self, batch):
    state = batch['full_state'] # BS x WINDOW x SIZE
    state = th.cat([th.zeros_like(state)[:,:1], state], dim=1)
    out, (h,c) = self.forward(state)
    delta = state[:,1:] - out[:,:-1]
    mse = (delta).pow(2).mean()
    return mse, {'mse': mse}

  def sample(self, n, action=None, prompts=None, prompt_n=10):
    # TODO: feed act_n
    with th.no_grad():
      if action is not None:
        n = action.shape[0]
      batch = {}
      if prompts is None:
        out, (h,c) = self.forward(th.zeros(n, 1, len(self.env.obs_keys)).to(self.device))
        imdts = [out]
        for i in range(self.window-1):
          out, (h,c) = self.forward(out, (h,c))
          imdts += [out]
        batch['full_state'] = th.cat(imdts, 1)
      else:
        state = prompts['full_state']
        state = th.cat([th.zeros_like(state)[:,:1], state], dim=1)
        out, (h, c) = self.forward(state[:,:5])
        imdts = [out]
        out = out[:,-1:]
        for i in range(4, self.window-1):
          out, (h, c) = self.forward(out, (h,c))
          imdts += [out]
        batch['full_state'] = th.cat(imdts, 1)
    return batch