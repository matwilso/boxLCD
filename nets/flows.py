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
import torch.distributions.transforms as tran
from torch import nn
import torch.nn.functional as F
from scipy.optimize import bisect

# def inverse_permute(idxs, y):
#  return torch.empty_like(idxs).scatter_(dim=1, index=idxs, src=y)

def make_logistic(a, b):
  base_distribution = tdib.Uniform(0, 1)
  transforms = [tdib.transforms.SigmoidTransform().inv, tdib.transforms.AffineTransform(loc=a, scale=b)]
  logistic = tdib.TransformedDistribution(base_distribution, transforms)
  return logistic

class MLP(nn.Module):
  def __init__(self, input_size, n_hidden, hidden_size, output_size):
    super().__init__()
    layers = []
    for _ in range(n_hidden):
      layers.append(nn.Linear(input_size, hidden_size))
      layers.append(nn.ReLU())
      input_size = hidden_size
    layers.append(nn.Linear(hidden_size, output_size))
    self.layers = nn.Sequential(*layers)

  def forward(self, x, cond=None):
    if cond is not None:
      x = torch.cat([x, cond], -1)
    return self.layers(x)

class Inverse(nn.Module):
  def __init__(self, base_flow):
    super().__init__()
    self.base_flow = base_flow

  def flow(self, x, **kwargs):
    return self.base_flow.invert(x, **kwargs)

  def invert(self, y, **kwargs):
    return self.base_flow.flow(y, **kwargs)

class Sigmoid(nn.Module):
  def flow(self, x, **kwargs):
    y = torch.sigmoid(x)
    logd = -F.softplus(x) - F.softplus(-x)
    return y, logd

  def invert(self, y, **kwargs):
    x = -torch.log(torch.reciprocal(y) - 1)
    logd = -y.log() - (1 - y).log()
    return x, logd

class ElemwiseAffine(nn.Module):
  def __init__(self, logscales, biases):
    super().__init__()
    self.biases = biases
    self.logscales = logscales

  def flow(self, x, **kwargs):
    return (x * torch.exp(self.logscales) + self.biases), self.logscales

  def invert(self, y, **kwargs):
    return ((y - self.biases) / torch.exp(self.logscales)), -self.logscales

class Compose(nn.Module):
  def __init__(self, flows):
    super().__init__()
    self.flows = flows

  def flow(self, x, **kwargs):
    bs = int((x[0] if isinstance(x, tuple) else x).shape[0])
    logd_terms = []
    for f in self.flows:
      x, l = f.flow(x, **kwargs)
      if l is not None:
        logd_terms.append(l)
    return x, sum(logd_terms) if logd_terms else torch.zeros(1)

  def invert(self, y, **kwargs):
    logd_terms = []
    for f in self.flows[::-1]:
      y, l = f.invert(y, **kwargs)
      if l is not None:
        logd_terms.append(l)
    return y, sum(logd_terms) if logd_terms else torch.zeros(1)


def _log_pdf(x, mean, log_scale):
  """Element-wise log density of the logistic distribution."""
  z = (x - mean) * torch.exp(-log_scale)
  log_p = z - log_scale - 2 * F.softplus(z)
  return log_p

def _log_cdf(x, mean, log_scale):
  """Element-wise log CDF of the logistic distribution."""
  z = (x - mean) * torch.exp(-log_scale)
  log_p = F.logsigmoid(z)
  return log_p

def mixture_log_pdf(x, prior_logits, means, log_scales):
  """Log PDF of a mixture of logistic distributions."""
  log_ps = F.log_softmax(prior_logits, dim=1) + _log_pdf(x.unsqueeze(1), means, log_scales)
  log_p = torch.logsumexp(log_ps, dim=1)
  return log_p

def mixture_log_cdf(x, prior_logits, means, log_scales):
  """Log CDF of a mixture of logistic distributions."""
  log_ps = F.log_softmax(prior_logits, dim=1) + _log_cdf(x.unsqueeze(1), means, log_scales)
  log_p = torch.logsumexp(log_ps, dim=1)
  return log_p


class MixtureCDFFlow(nn.Module):
  def __init__(self, logits, mu, logstd):
    super().__init__()
    self.cat = tdib.Categorical(logits=logits)
    self.logits = logits
    self.mu = mu
    self.logstd = logstd
    #self.component = tdib.MultivariateNormal(mu, torch.diag_embed(std))
    #self.mixture_dist = tdib.MixtureSameFamily(self.cat, self.component)
    self.mixture_dist = tdib.Normal
    self.n_components = logits.shape[-1]

  def flow(self, x):
    # parameters of flow on x depend on what it's conditioned on
    weights = F.softmax(self.logits, dim=-1)
    # z = cdf(x)
    dist = self.mixture_dist(self.mu, self.logstd.exp())
    bigx = x[:, :, None].repeat(1, 1, self.n_components, 1)
    z = (dist.cdf(bigx) * weights[..., None]).sum(dim=2)
    # log_det = log dz/dx = log pdf(x)
    log_det = (dist.log_prob(bigx).exp() * weights[..., None]).sum(dim=2).log()
    return z, log_det

  def invert(self, z, tol=1e-12):
    z = torch.clamp(z, 0.0, 1.0)
    with torch.no_grad():
      def bisect_iter(x, lb, ub):
        cur_z = self.flow(x)[0]
        gt = (cur_z > z).type(z.dtype)
        lt = 1 - gt
        new_x = gt * (x + lb) / 2. + lt * (x + ub) / 2.
        new_lb = gt * lb + lt * x
        new_ub = gt * x + lt * ub
        diff = (new_x - x).abs().max()
        return new_x, new_lb, new_ub, diff

      x = torch.zeros_like(z)
      maxscales = self.logstd.exp().sum(-2)[..., None, :]
      lb = torch.min(self.mu - 10 * maxscales, dim=-2)[0]
      ub = torch.max(self.mu + 10 * maxscales, dim=-2)[0]
      diff = 1e9

      for i in range(200):
        if diff <= tol:
          break
        x, lb, ub, diff = bisect_iter(x, lb, ub)
    return x, -self.flow(x)[1]

  def log_prob(self, x):
    z, log_det = self.flow(x)
    return self.base_dist.log_prob(z) + log_det

class MixtureCDFFLowCoupling(nn.Module):
  def __init__(self, shape, mask_idxs, cond_size=0, n_hidden=2, hidden_size=256, C=None):
    super().__init__()
    self.C = C
    self.mask = self.build_mask(shape, mask_idxs)
    self.cond_size = cond_size
    self.shape = shape
    K = C.mdn_k
    outshape = K + 2 * K * shape  # MDN
    outshape += 2 * shape  # logscales and biases

    self.mlp = MLP(input_size=shape + cond_size, n_hidden=n_hidden, hidden_size=hidden_size, output_size=outshape)
    self.mlp.to(C.device)

  def build_mask(self, shape, idxs):
    mask = torch.zeros(shape).to(self.C.device)
    mask[idxs] = 1.0
    return mask

  def forward(self, x, cond=None, reverse=False):
    # returns transform(x), log_det
    mask = torch.ones(x.shape[:2] + (1,), device=x.device) * self.mask
    x_ = x * mask
    params = self.mlp(x_, cond)

    scalebias, mdn_params = params[..., :2 * self.shape], params[..., 2 * self.shape:]
    logscales, biases = scalebias.chunk(2, -1)
    K = self.C.mdn_k
    Kx = K * self.shape
    logits, mu, logstd = mdn_params[..., :K], mdn_params[..., K:K + Kx], mdn_params[..., K + Kx:]
    mu = mu.reshape(mu.shape[:-1] + (-1, self.shape))
    logstd = logstd.reshape(logstd.shape[:-1] + (-1, self.shape))

    flow_stack = Compose([
        MixtureCDFFlow(logits, mu, logstd),
        Inverse(Sigmoid()),
        ElemwiseAffine(logscales=logscales, biases=biases)
    ])
    if reverse:  # inverting the transformation
      out, log_det = flow_stack.invert(x)
    else:
      out, log_det = flow_stack.flow(x)

    x = (1 - mask) * out + x_
    log_det = log_det * (1.0 - mask)
    return x, log_det

class RealNVP(nn.Module):
  def __init__(self, shape, transforms, cond=None, C=None):
    super().__init__()
    self.shape = shape
    #self.prior = torch.distributions.MultivariateNormal(torch.zeros(shape).to(C.device), torch.diag_embed(torch.ones(shape).to(C.device)))
    #self.prior = torch.distributions.Uniform(torch.tensor(-1.).to(C.device), torch.tensor(1.).to(C.device))
    self.prior = torch.distributions.Uniform(torch.tensor(0.).to(C.device), torch.tensor(1.).to(C.device))
    #self.prior = torch.distributions.Normal(torch.tensor(0.).to(C.device), torch.tensor(1.).to(C.device))
    self.transforms = nn.ModuleList(transforms)
    self.cond = cond

  @property
  def mean(self):
    z = self.prior.mean * torch.ones([*self.cond.shape[:2], self.shape]).to(self.C.device)
    return self.invert_flow(z)

  def flow(self, x):
    # maps x -> z, and returns the log determinant (not reduced)
    z, log_det = x, torch.zeros_like(x)
    for op in self.transforms:
      z, delta_log_det = op.forward(z, cond=self.cond)
      log_det += delta_log_det
    return z, log_det

  def invert_flow(self, z):
    # z -> x (inverse of f)
    for op in reversed(self.transforms):
      z, _ = op.forward(z, cond=self.cond, reverse=True)
    return z

  def log_prob(self, x):
    z, log_det = self.flow(x)
    return torch.sum(log_det, dim=1) + torch.sum(self.prior.log_prob(z), dim=1)

  def sample(self):
    z = self.prior.sample([*self.cond.shape[:2], self.shape])
    return self.invert_flow(z)

  def nll(self, x):
    return -self.log_prob(x).mean()


class AffineTransform(nn.Module):
  def __init__(self, shape, mask_idxs, cond_size=0, n_hidden=2, hidden_size=256, C=None):
    super().__init__()
    self.C = C
    self.mask = self.build_mask(shape, mask_idxs)
    self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
    self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
    self.cond_size = cond_size
    self.mlp = MLP(input_size=shape + cond_size, n_hidden=n_hidden, hidden_size=hidden_size, output_size=2 * shape)
    self.mlp.to(C.device)

  def build_mask(self, shape, idxs):
    mask = torch.zeros(shape).to(self.C.device)
    mask[idxs] = 1.0
    return mask

  def forward(self, x, cond=None, reverse=False):
    # returns transform(x), log_det
    mask = torch.ones(x.shape[:2] + (1,), device=x.device) * self.mask
    x_ = x * mask
    log_s, t = self.mlp(x_, cond).chunk(2, dim=-1)
    log_s = self.scale * torch.tanh(log_s) + self.scale_shift
    t = t * (1.0 - mask)
    log_s = log_s * (1.0 - mask)

    if reverse:  # inverting the transformation
      x = (x - t) * torch.exp(-log_s)
    else:
      x = x * torch.exp(log_s) + t
    return x, log_s
