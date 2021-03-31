import numpy as np
import torch as th
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributions as thd
from torch.optim import Adam
import utils

class SVAE(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    H = C.hidden_size
    # encoder -> VQ -> decoder
    self.encoder = Encoder(env, C)
    self.decoder = Decoder(env, C)
    self.optimizer = Adam(self.parameters(), C.lr)

    self.vq = Quantize(C.hidden_size, C.vqK)
    _ = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10e3, eta_min=1/16, last_epoch=-1, verbose=False)
    Tmax = 10e3
    nMax = 1.0
    nMin = 1/16
    def cos_schedule(t):
      x = np.clip(t/Tmax, 0.0, 1.0)
      return nMin + 0.5*(nMax-nMin) * (1 + np.cos(x * np.pi))
    #self.cos_schedule = cos_schedule
    self.env = env
    self.C = C
  
  def save(self, dir):
    print("SAVED MODEL", dir)
    path = dir / 'svae.pt'
    sd = self.state_dict()
    sd['C'] = self.C
    th.save(sd, path)
    print(path)

  def load(self, dir):
    path = dir / 'svae.pt'
    sd = th.load(path)
    C = sd.pop('C')
    self.load_state_dict(sd)
    print(f'LOADED {path}')

  def train_step(self, batch, dry=False):
    if dry:
      return {}
    #temp = self.cos_schedule(self.optimizer._step_count)
    temp = 1.0
    self.vq.temperature = temp
    self.optimizer.zero_grad()
    flatter_batch = {key: val.flatten(0, 1) for key, val in batch.items()}
    loss, metrics = self.loss(flatter_batch)
    metrics['temp'] = th.as_tensor(temp)
    if not dry:
      loss.backward()
      self.optimizer.step()
    return metrics

  def loss(self, batch, eval=False, return_idxs=False):
    z_q, diff, idxs, decoded = self.forward(batch)
    pstate_loss = -decoded.log_prob(batch['pstate']).mean()
    loss = pstate_loss + diff
    metrics = {'total_loss': loss, 'pstate_loss': pstate_loss, 'entropy_loss': diff, 'pstate_delta': ((decoded.mean-batch['pstate'])**2).mean()}
    metrics['zqdelta'] = (z_q-idxs).abs().mean()
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
    flatter_batch = {key: val[:8,0] for key, val in batch.items()}
    z_q, diff, idxs, decoded = self.forward(flatter_batch)
    pred_pstate = decoded.mean.cpu().numpy()
    true_pstate = flatter_batch['pstate'].cpu().numpy()
    true_pstate_imgs = []
    pred_pstate_imgs = []
    for i in range(8):
      true_pstate_imgs += [self.env.reset(pstate=true_pstate[i])['lcd']]
      pred_pstate_imgs += [self.env.reset(pstate=pred_pstate[i])['lcd']]
    out = np.concatenate([np.stack(true_pstate_imgs), np.stack(pred_pstate_imgs)], 0)[:,None]
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
    H = C.hidden_size
    state_n = env.observation_space.spaces['pstate'].shape[0]
    self.state_embed = nn.Sequential(
        nn.Linear(state_n, H),
        nn.ReLU(),
        nn.Linear(H, H),
        nn.ReLU(),
        nn.Linear(H, H),
    )

  def forward(self, batch):
    state = batch['pstate']
    x = self.state_embed(state)
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
  def forward(self, x):
    return thd.Normal(self.state_net(x), 1)

class Quantize(nn.Module):
  """there is no god"""
  def __init__(self, num_hiddens, n_embed):
    super().__init__()
    self.n_embed = n_embed
    self.temperature = 1.0
    self.kld_scale = 5e-4
    self.proj = nn.Linear(num_hiddens, n_embed)

  def forward(self, z):
    logits = self.proj(z)
    dist = thd.Bernoulli(logits=logits)
    z_q = dist.sample()
    z_q += dist.probs - dist.probs.detach()
    # + kl divergence to the prior loss (entropy bonus)
    diff = self.kld_scale * dist.entropy().mean()
    return z_q, diff, z_q

  #def forward(self, z):
  #  # force hard = True when we are in eval mode, as we must quantize
  #  hard = self.straight_through if self.training else True

  #  logits = self.proj(z)
  #  soft_bin = thd.RelaxedBernoulli(self.temperature, logits=logits)
  #  z_q = soft_bin.rsample()

  #  hard_z_q = z_q.round()
  #  if hard:
  #    z_q = hard_z_q - z_q.detach() + z_q

  #  # + kl divergence to the prior loss (entropy bonus)
  #  diff = self.kld_scale * thd.Bernoulli(logits=logits).entropy().mean()
  #  return z_q, diff, hard_z_q