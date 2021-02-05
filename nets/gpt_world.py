import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F
from nets.common import GaussHead, MDNHead, CausalSelfAttention, Block, BinaryHead, CustomHead, CustomEmbed, FlowHead
from nets.gpt import GPT

class GPTDist(nn.Module):
  def __init__(self, gpt, x):
    super().__init__()
    self.gpt = gpt
    self.x = x
  def log_prob(self, state):
    x = state.flatten(0,1)[...,None]
    dist = self.gpt.forward(x, cond=self.x.flatten(0,1))
    return dist.log_prob(x)
  def sample(self, prompts=None):
    shape = self.x.shape
    return self.gpt.sample(np.prod(shape[:2]), cond=self.x.flatten(0,1), prompts=prompts)[0].reshape(shape[:2] + (-1,))

class MultiHead(nn.Module):
  def __init__(self, in_n, out_n, split, C):
    super().__init__()
    self.C = C
    self.in_n = in_n
    self.out_n = out_n
    self.split = split
    self.layer = nn.Linear(in_n, in_n * 2)
    self.binary = BinaryHead(in_n, self.split, C)
    #self.state = FlowHead(in_n, out_n - self.split, C)
    #self.state = GaussHead(in_n, out_n-self.split, C) 
    self.state = MDNHead(in_n, out_n-self.split, C) 
    #self.state = GPT(1, block_size=out_n-self.split, dist='mdn', cond=in_n, C=C) 

  def forward(self, x, past_o=None):
    xb, xs = self.layer(x).chunk(2, -1)
    bin = self.binary(xb) if 'image' in self.C.subset else None
    state = self.state(xs) if 'state' in self.C.subset else None
    #return bin, GPTDist(self.state, xs)
    return bin, state

class GPTWorld(nn.Module):
  def __init__(self, size, block_size, dist, cond=None, C=None):
    super().__init__()
    self.C = C
    self.size = size
    self.block_size = block_size
    self.cond = cond
    self.dist = dist
    self.imsize = self.C.lcd_h*self.C.lcd_w
    self.pos_emb = nn.Parameter(torch.zeros(1, block_size, C.n_embed))
    if self.cond is not None:
      self.cond_in = nn.Linear(self.cond, C.n_embed//2, bias=False)
    # input embedding stem
    self.embed = nn.Linear(size, C.n_embed//2 if cond is not None else C.n_embed, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[Block(block_size, C) for _ in range(C.n_layer)])
    # decoder head
    self.ln_f = nn.LayerNorm(C.n_embed)
    if dist == 'gauss':
      self.dist_head = GaussHead(self.size, C)
    elif dist == 'mdn':
      self.dist_head = MDNHead(self.size, C)
    elif dist == 'binary':
      self.dist_head = BinaryHead(C.n_embed, self.size, C)
    elif dist == 'custom':
      self.dist_head = CustomHead(C.n_embed, self.size, C)
      self.custom_embed = CustomEmbed(size, C.n_embed//2 if cond is not None else C.n_embed, C)
    elif dist == 'multi':
      self.dist_head = MultiHead(C.n_embed, C.state_n+self.imsize, self.imsize, C)
    self.to(C.device)

  def append_location(self, x):
    """add loc coords to every elem"""
    X = torch.linspace(-1, 1, x.shape[-2])
    return torch.cat([x, X[None, ..., None].repeat_interleave(x.shape[0], 0).to(x.device)], -1)

  def forward(self, x, cond=None):
    BS, T, E = x.shape
    # SHIFT RIGHT (add a padding on the left)
    x = torch.cat([torch.zeros(BS, 1, E).to(self.C.device), x[:, :-1]], dim=1)
    #x = self.append_location(x)
    # forward the GPT model
    if self.dist == 'custom':
      x = self.custom_embed(x)

    x = self.embed(x)
    if self.cond is not None:
      cin = self.cond_in(cond)
      if cond.ndim == 2:
        x = torch.cat([x, cin[:,None].repeat_interleave(self.block_size, 1)], -1)
      else:
        x = torch.cat([x, cin], -1)
    x += self.pos_emb # each position maps to a (learnable) vector

    # add padding on left so that we can't see ourself.
    x = self.blocks(x)
    logits = self.ln_f(x)
    # TODO: probably return logits as well.
    # probably just logits and move loss and stuff out of here
    return self.dist_head(logits)

  def loss(self, x, cond=None):
    dist = self.forward(x, cond)
    import ipdb; ipdb.set_trace()
    return -dist.log_prob(x), dist

  def sample(self, n, cond=None, prompts=None):
    # TODO: feed act_n
    with torch.no_grad():
      if cond is not None:
        n = cond.shape[0]
      samples = torch.zeros(n, self.block_size, self.imsize+self.C.state_n).to(self.C.device)
      start = 0
      if prompts is not None:
        import ipdb; ipdb.set_trace()
        lcd = prompts['lcd'].flatten(-2).type(samples.dtype)
        samples[:, :5, :self.imsize] = lcd
        start = lcd.shape[1]-1
      #acts = (torch.rand(samples.shape[:-1]) * 2 - 1).to(self.C.device)[..., None]
      for i in range(start, self.block_size-1):
        # TODO: check this setting since we have modified things
        bindist, statedist = self.forward(samples, cond)
        samples[:, i + 1, :self.imsize] = bindist.sample()[:,i]
        import ipdb; ipdb.set_trace()
        samples[:, i + 1, self.imsize:] = statedist.sample()[:,i]
        if i == self.block_size-2:
          logp = bindist.log_prob(samples[...,:self.imsize])

    return samples, logp.mean().item()

