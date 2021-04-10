import numpy as np
import torch as th
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributions as thd
from torch.optim import Adam
import utils

class QVAE(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    # encoder -> VQ -> decoder
    self.encoder = Encoder(env, C)
    self.decoder = Decoder(env, C)
    print(C.hidden_size, C.vqK)
    self.vq = Quantize(C.hidden_size, C.vqK)
    self.optimizer = Adam(self.parameters(), C.lr)
    self.env = env
    self.C = C

  def save(self, dir):
    print("SAVED MODEL", dir)
    path = dir / 'qvae.pt'
    sd = self.state_dict()
    sd['C'] = self.C
    th.save(sd, path)
    print(path)

  def load(self, dir):
    path = dir / 'qvae.pt'
    sd = th.load(path)
    C = sd.pop('C')
    self.load_state_dict(sd)
    print(f'LOADED {path}')

  def train_step(self, batch, dry=False):
    if dry:
      return {}
    self.optimizer.zero_grad()
    flatter_batch = {key: val.flatten(0, 1) for key, val in batch.items()}
    loss, metrics = self.loss(flatter_batch)
    if not dry:
      loss.backward()
      self.optimizer.step()
    return metrics

  def loss(self, batch, eval=False, return_idxs=False):
    z_q, diff, idxs, decoded = self.forward(batch)
    image_loss = -decoded['lcd'].log_prob(batch['lcd']).mean()
    pstate_loss = -decoded['pstate'].log_prob(batch['pstate']).mean()
    loss = image_loss + pstate_loss + diff
    metrics = {'total_loss': loss, 'image_loss': image_loss, 'pstate_loss': pstate_loss, 'entropy_loss': diff}
    metrics['zqdelta'] = (z_q - idxs).abs().mean()
    if eval:
      metrics['decoded'] = decoded
    if return_idxs:
      metrics['idxs'] = idxs
    return loss, metrics

  def forward(self, x):
    z_e = self.encoder(x)
    z_q, diff, idxs = self.vq(z_e)
    decoded = self.decoder(z_q)
    return z_q, diff, idxs, decoded

  def evaluate(self, writer, batch, epoch):
    self.vq.eval()
    flatter_batch = {key: val[:8, 0] for key, val in batch.items()}
    z_q, diff, idxs, decoded = self.forward(flatter_batch)
    image = decoded['lcd'].sample()
    true_image = flatter_batch['lcd'][:, None].cpu()
    image = th.cat([true_image, image.cpu()], 0)
    writer.add_image('recon_image', utils.combine_imgs(image, 2, 8)[None], epoch)

    pred_pstate = decoded['pstate'].mean.cpu().numpy()
    true_pstate = flatter_batch['pstate'].cpu().numpy()
    true_pstate_imgs = []
    pred_pstate_imgs = []
    for i in range(8):
      true_pstate_imgs += [self.env.reset(pstate=true_pstate[i])['lcd']]
      pred_pstate_imgs += [self.env.reset(pstate=pred_pstate[i])['lcd']]
    out = np.concatenate([np.stack(true_pstate_imgs), np.stack(pred_pstate_imgs)], 0)[:, None]
    writer.add_image('recon_pstate', utils.combine_imgs(out, 2, 8)[None], epoch)
    self.vq.train()

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
    H = C.nfilter
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
        #ResBlock(H, emb_channels=H),
        nn.Conv2d(H, H, 3, 2, padding=1),
        #ResBlock(H, emb_channels=H),
        nn.Conv2d(H, H, 3, 2, padding=1),
        #ResBlock(H, emb_channels=H),
        nn.Conv2d(H, H // 8, 1),
        nn.Flatten(-3),
        nn.Linear(H, C.hidden_size)
    ])

  def forward(self, batch):
    state = batch['pstate']
    lcd = batch['lcd']
    emb = self.state_embed(state)
    x = lcd[:, None]
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
      nn.Linear(C.vqK, H),
      nn.ReLU(),
      nn.Linear(H, H),
      nn.ReLU(),
      nn.Linear(H, state_n),
    )
    self.emb = nn.Linear(C.vqK, C.hidden_size)

    H = C.nfilter
    assert C.lcd_h == 16, C.lcd_w == 32
    self.net = nn.Sequential(
      nn.Conv2d(1, H, 3, 1, padding=1),
      nn.ReLU(),
      ResBlock(H, emb_channels=C.hidden_size),
      nn.Conv2d(H, H, 3, 1, padding=1),
      nn.ReLU(),
      ResBlock(H, emb_channels=C.hidden_size),
      nn.Conv2d(H, 1, 3, 1, padding=1),
    )
    self.C = C

  def forward(self, x):
    img = x.reshape([-1, 1, 16, 32])
    emb = self.emb(x)
    for layer in self.net:
      if isinstance(layer, ResBlock):
        img = layer(img, emb)
      else:
        img = layer(img)
    lcd_dist = thd.Bernoulli(logits=img)
    #lcd_dist = thd.Bernoulli(logits=self.net(img))
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

class Quantize(nn.Module):
  """there is no god"""
  def __init__(self, num_hiddens, n_embed):
    super().__init__()
    self.n_embed = n_embed
    self.kld_scale = 5e-4
    self.proj = nn.Linear(num_hiddens, n_embed)

  def forward(self, z):
    logits = self.proj(z)
    dist = thd.Bernoulli(logits=logits)
    z_q = dist.sample()
    z_q += dist.probs - dist.probs.detach() # straight-through gradient
    # + kl divergence to the prior loss (entropy bonus)
    diff = self.kld_scale * dist.entropy().mean()
    return z_q, diff, z_q
