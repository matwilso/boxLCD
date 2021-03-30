import torch as th
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributions as thd
from torch.optim import Adam
import utils

class TVQVAE(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    # encoder -> VQ -> decoder
    self.encoder = Encoder(env, C)
    self.vq = GumbelQuantize(128, C.vqK, C.vqD)
    self.decoder = Decoder(env, C)
    self.optimizer = Adam(self.parameters(), C.lr)
    self.C = C

  def train_step(self, batch, dry=False):
    self.optimizer.zero_grad()
    flatter_batch = {key: val.flatten(0, 1) for key, val in batch.items()}
    loss, metrics = self.loss(flatter_batch)
    if not dry:
      loss.backward()
      self.optimizer.step()
    return metrics

  def loss(self, batch, eval=False, return_idxs=False):
    embed_loss, decoded, perplexity, idxs = self.forward(batch)
    import ipdb; ipdb.set_trace()

    recon_loss = -thd.Bernoulli(logits=decoded).log_prob(x).mean()
    loss = recon_loss + embed_loss
    prior_loss = th.zeros(1)
    metrics = {'vq_vae_loss': loss, 'recon_loss': recon_loss, 'embed_loss': embed_loss, 'perplexity': perplexity, 'prior_loss': prior_loss}
    if eval:
      metrics['decoded'] = decoded
    if return_idxs:
      metrics['idxs'] = idxs
    return loss, metrics

  def forward(self, x):
    z_e = self.encoder(x)
    z_q, diff, idxs = self.vq(z_e)
    import ipdb; ipdb.set_trace()
    decoded = self.decoder(z_q)
    return z_q, diff, idxs

  def sample(self, n):
    import ipdb; ipdb.set_trace()
    prior_idxs = self.transformerCNN.sample(n)[0]
    prior_enc = self.vq.idx_to_encoding(prior_idxs)
    prior_enc = prior_enc.reshape([n, 7, 7, -1]).permute(0, 3, 1, 2)
    decoded = self.decoder(prior_enc)
    return 1.0 * (decoded.exp() > 0.5).cpu()

class Encoder(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    state_n = env.observation_space.spaces['pstate'].shape[0]
    self.state_embed = nn.Sequential(
        nn.Linear(state_n, H),
        nn.ReLU(),
        nn.Linear(H, H),
        nn.ReLU(),
        nn.Linear(H, H),
    )
    self.seq = nn.ModuleList([
        nn.Conv2d(1, H, 3, 2, padding=1),
        ResBlock(H, emb_channels=H),
        nn.Conv2d(H, H, 3, 2, padding=1),
        ResBlock(H, emb_channels=H),
        nn.Conv2d(H, H, 3, 2, padding=1),
        ResBlock(H, emb_channels=H),
        nn.Conv2d(H, H//8, 1),
        nn.Flatten(-3),
        nn.Linear(H, H)
    ])

  def forward(self, batch):
    state = batch['pstate']
    lcd = batch['lcd']
    emb = self.state_embed(state)
    x = lcd[:,None]
    for layer in self.seq:
      if isinstance(layer, ResBlock):
        x = layer(x, emb)
      else:
        x = layer(x)
    return x

class Upsample(nn.Module):
  """double the size of the input"""
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
  def forward(self, x, emb=None):
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = self.conv(x)
    return x

class Decoder(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    state_n = env.observation_space.spaces['pstate'].shape[0]
    self.state_net = nn.Sequential(
        nn.Flatten(-3),
        nn.Linear(C.vqD * 4 * 6, H * C.vidstack),
        nn.Unflatten(-1, (C.vidstack, H)),
        nn.ReLU(),
        nn.Linear(H, H),
        nn.ReLU(),
        nn.Linear(H, state_n),
    )

    self.net = nn.Sequential(
        Upsample(C.vqD, H),
        nn.ReLU(),
        Upsample(H, H),
        nn.ReLU(),
        nn.Conv2d(H, H, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(H, C.vidstack, 3, padding=1),
    )
  def forward(self, x):
    lcd_dist = thd.Bernoulli(logits=self.net(x))
    state_dist = thd.Normal(self.state_net(x), 1)
    return {'lcd': lcd_dist, 'pstate': state_dist}

class ResBlock(nn.Module):
  def __init__(self, channels, emb_channels, out_channels=None, dropout=0.0):
    super().__init__()
    self.out_channels = out_channels or channels

    self.in_layers = nn.Sequential(
        nn.GroupNorm(32, channels),
        nn.SiLU(),
        nn.Conv2d(channels, self.out_channels, 3, padding=1)
    )
    self.emb_layers = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_channels, self.out_channels)
    )
    self.out_layers = nn.Sequential(
        nn.GroupNorm(32, self.out_channels),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        utils.zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
    )
    if self.out_channels == channels:
      self.skip_connection = nn.Identity()
    else:
      self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)  # step down size

  def forward(self, x, emb):
    h = self.in_layers(x)
    emb_out = self.emb_layers(emb)[..., None, None]
    h = h + emb_out
    h = self.out_layers(h)
    return self.skip_connection(x) + h


class GumbelQuantize(nn.Module):
  """
  Gumbel Softmax trick quantizer
  Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
  https://arxiv.org/abs/1611.01144
  """
  def __init__(self, num_hiddens, n_embed, embedding_dim, straight_through=False):
    super().__init__()

    self.embedding_dim = embedding_dim
    self.n_embed = n_embed

    self.straight_through = straight_through
    self.temperature = 1.0
    self.kld_scale = 5e-4

    self.proj = nn.Linear(num_hiddens, n_embed)
    self.embed = nn.Embedding(1, embedding_dim)

  def forward(self, z):
    # force hard = True when we are in eval mode, as we must quantize
    hard = self.straight_through if self.training else True

    logits = self.proj(z)
    soft_bin = thd.RelaxedBernoulli(self.temperature, logits=logits)
    z_q = soft_bin.rsample()

    # + kl divergence to the prior loss
    # entropy bonus
    diff = self.kld_scale * thd.Bernoulli(logits=logits).entropy().mean()
    idxs = z_q > 0.5
    return z_q, diff, idxs