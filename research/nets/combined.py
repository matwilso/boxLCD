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

class Combined(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    #self.vqvae = VQVAE(env, C)
    self.state_vqvae = State_VQVAE(env, C)

  def loss(self, x, eval=False):
    iloss, imetrics = self.vqvae.loss(x, eval=eval)
    #sloss, smetrics = self.state_vqvae.loss(x, eval=eval)
    return sloss, smetrics

  def forward(self, x):
    import ipdb; ipdb.set_trace()