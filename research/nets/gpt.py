import numpy as np
from torch.optim import Adam
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from .common import BinaryHead, CategoricalHead


class GPT(nn.Module):
  """  the full GPT language model, with a context size of block_size """
  def __init__(self, in_size, block_size, head='cat', cond_size=None, G=None):
    super().__init__()
    assert G is not None, 'must pass in G'
    self.block_size = block_size
    self.in_size = in_size
    self.pos_emb = nn.Parameter(th.zeros(1, self.block_size, G.n_embed)) # learned position embedding
    self.embed = nn.Linear(self.in_size, G.n_embed, bias=False)
    self.blocks = nn.Sequential(*[TransformerBlock(self.block_size, G) for _ in range(G.n_layer)])
    self.ln_f = nn.LayerNorm(G.n_embed)
    if head == 'bin':
      self.dist_head = BinaryHead(G.n_embed, self.in_size, G)
    elif head == 'cat':
      self.dist_head = CategoricalHead(G.n_embed, self.in_size, G)

    self.cond_size = cond_size
    if cond_size is not None:
      self.cond_in = nn.Sequential(
        nn.Linear(self.cond_size, G.n_embed),
        nn.ReLU(),
        nn.Linear(G.n_embed, G.n_embed, bias=False),
      )
    self.G = G

  def forward(self, x, cond=None):
    BS, T, G = x.shape
    # SHIFT RIGHT (add a padding on the left) so you can't see yourself 
    x = th.cat([th.zeros(BS, 1, G).to(self.G.device), x[:, :-1]], dim=1)
    # forward the GPT model
    x = self.embed(x)
    x += self.pos_emb # each position maps to a (learnable) vector
    if cond is not None:
      cond = self.cond_in(cond)
      cond = th.cat([th.zeros(BS, 1, self.G.n_embed).to(self.G.device), cond[:, :-1]], dim=1)
      x += cond
    # add padding on left so that we can't see ourself.
    x = self.blocks(x)
    logits = self.ln_f(x)
    return self.dist_head(logits)

  def sample(self, n):
    steps = []
    batch = th.zeros(n, self.block_size, self.in_size).to(self.G.device)
    for i in range(self.block_size):
      dist = self.forward(batch)
      batch[:,i] = dist.sample()[:,i]
      steps += [batch.cpu()]
    return batch, steps

class CausalSelfAttention(nn.Module):
  """
  A vanilla multi-head masked self-attention layer with a projection at the end.
  It is possible to use th.nn.MultiheadAttention here but I am including an
  explicit implementation here to show that there is nothing too scary here.
  """
  def __init__(self, block_size, G):
    super().__init__()
    self.block_size = block_size
    assert G.n_embed % G.n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(G.n_embed, G.n_embed)
    self.query = nn.Linear(G.n_embed, G.n_embed)
    self.value = nn.Linear(G.n_embed, G.n_embed)
    # output projection
    self.proj = nn.Linear(G.n_embed, G.n_embed)
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", th.tril(th.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size))
    self.n_head = G.n_head

  def forward(self, x, layer_past=None):
    B, T, G = x.size()
    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    k = self.key(x).view(B, T, self.n_head, G // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.n_head, G // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.n_head, G // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
    att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, G)  # re-assemble all head outputs side by side
    # output projection
    y = self.proj(y)
    return y

class TransformerBlock(nn.Module):
  """ an unassuming Transformer block """
  def __init__(self, block_size, G):
    super().__init__()
    self.ln1 = nn.LayerNorm(G.n_embed)
    self.ln2 = nn.LayerNorm(G.n_embed)
    self.attn = CausalSelfAttention(block_size, G)
    self.mlp = nn.Sequential(
        nn.Linear(G.n_embed, 4 * G.n_embed),
        nn.GELU(),
        nn.Linear(4 * G.n_embed, G.n_embed),
    )
  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x