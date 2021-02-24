import sys
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import torchvision
from torch.optim import Adam
from itertools import chain, count
import torch
from torch import distributions as tdib
from torch import nn
import torch.nn.functional as F

# GPT IMPLEMENTATION TAKEN FROM https://github.com/karpathy/minGPT AND THEN HACKED TO BITS

class CausalSelfAttention(nn.Module):
  """
  A vanilla multi-head masked self-attention layer with a projection at the end.
  It is possible to use torch.nn.MultiheadAttention here but I am including an
  explicit implementation here to show that there is nothing too scary here.
  """

  def __init__(self, block_size, C):
    super().__init__()
    self.block_size = block_size
    assert C.n_embed % C.n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(C.n_embed, C.n_embed)
    self.query = nn.Linear(C.n_embed, C.n_embed)
    self.value = nn.Linear(C.n_embed, C.n_embed)
    # output projection
    self.proj = nn.Linear(C.n_embed, C.n_embed)
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size))
    self.C = C

  def forward(self, x, layer_past=None):
    B, T, C = x.size()
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    k = self.key(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.C.n_head, C // self.C.n_head).transpose(1, 2)  # (B, nh, T, hs)
    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
    att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
    # output projection
    y = self.proj(y)
    return y

class Block(nn.Module):
  """ an unassuming Transformer block """

  def __init__(self, block_size, C):
    super().__init__()
    self.ln1 = nn.LayerNorm(C.n_embed)
    self.ln2 = nn.LayerNorm(C.n_embed)
    self.attn = CausalSelfAttention(block_size, C)
    self.mlp = nn.Sequential(
        nn.Linear(C.n_embed, 4 * C.n_embed),
        nn.GELU(),
        nn.Linear(4 * C.n_embed, C.n_embed),
    )

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x

class GPT(nn.Module):
  def __init__(self, act_dim, C):
    super().__init__()
    self.C = C
    self.imsize = self.C.lcd_h * self.C.lcd_w
    self.block_size = self.C.ep_len
    self.size = self.imsize
    # embedding
    self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, C.n_embed))
    self.act_condition = nn.Linear(act_dim, C.n_embed//2, bias=False)
    self.embed = nn.Linear(self.size, C.n_embed//2, bias=False)
    # transformer
    self.blocks = nn.Sequential(*[Block(self.block_size, C) for _ in range(C.n_layer)])
    # decoder head distributipn
    self.ln_f = nn.LayerNorm(C.n_embed)
    self.dist_head = BinaryHead(C.n_embed, self.size, C)
    self.to(C.device)

  def append_location(self, x):
    """add loc coords to every elem"""
    X = torch.linspace(-1, 1, x.shape[-2])
    return torch.cat([x, X[None, ..., None].repeat_interleave(x.shape[0], 0).to(x.device)], -1)

  def forward(self, batch):
    BS, LEN, *HW = batch['lcd'].shape
    E = np.prod(HW)
    x = batch['lcd'].reshape(BS, LEN, E) # flatten lcd to make a single flat frame a token
    acts = batch['acts']

    # SHIFT RIGHT (add a padding on the left) so you can't see yourself 
    x = torch.cat([torch.zeros(BS, 1, E).to(self.C.device), x[:, :-1]], dim=1)
    # forward the GPT model
    x = self.embed(x)
    cin = self.act_condition(acts)
    if acts.ndim == 2:
      x = torch.cat([x, cin[:,None].repeat_interleave(self.block_size, 1)], -1)
    else:
      x = torch.cat([x, cin], -1)
    x += self.pos_emb # each position maps to a (learnable) vector

    # add padding on left so that we can't see ourself.
    x = self.blocks(x)
    logits = self.ln_f(x)
    return self.dist_head(logits)

  def loss(self, batch):
    dist = self.forward(batch)
    lcd_loss = -dist.log_prob(batch['lcd'].reshape(dist.logits.shape)).mean()
    return lcd_loss

  def sample(self, n, acts=None, prompts=None):
    # TODO: feed act_n
    with torch.no_grad():
      if acts is not None:
        n = acts.shape[0]
      batch = {}
      batch['lcd'] = torch.zeros(n, self.block_size, self.imsize).to(self.C.device)
      batch['acts'] = acts if acts is not None else (torch.rand(n, self.block_size, self.act_n) * 2 - 1).to(self.C.device)
      start = 0
      if prompts is not None:
        lcd = prompts['lcd'].flatten(-2).type(batch['lcd'].dtype)
        batch['lcd'][:, :10] = lcd
        start = lcd.shape[1]
      for i in range(start, self.block_size):
        bindist = self.forward(batch)
        batch['lcd'][:, i] = bindist.sample()[:, i]
        if i == self.block_size - 1:
          sample_loss = self.loss(batch)
    batch['lcd'] = batch['lcd'].reshape(n, -1, 1, self.C.lcd_h, self.C.lcd_w)
    return batch, sample_loss.mean().cpu().detach()

class BinaryHead(nn.Module):
  """take logits and produce a bernoulli distribution independently on each element of the token"""
  def __init__(self, in_n, out_n, C):
    super().__init__()
    self.C = C
    self.layer = nn.Linear(in_n, out_n)

  def forward(self, x, past_o=None):
    x = self.layer(x)
    return tdib.Bernoulli(logits=x)