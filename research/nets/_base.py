import torch as th
from torch import nn
from torch.optim import Adam
import ignite

class Net(nn.Module):
  def __init__(self, G):
    super().__init__()
    self.G = G
    self.name = self.__class__.__name__
    self.ssim = ignite.metrics.SSIM(1.0, device=self.G.device)
    self.psnr = ignite.metrics.PSNR(1.0, device=self.G.device)
    self.cossim = nn.CosineSimilarity(dim=-1)

  def _init(self):
    self.optimizer = Adam(self.parameters(), lr=self.G.lr)

  def train_step(self, batch, dry=False):
    self.optimizer.zero_grad()
    loss, metrics = self.loss(batch)
    if not dry:
      loss.backward()
      self.optimizer.step()
    return metrics

  def save(self, dir):
    print("SAVED MODEL", dir)
    path = dir / f'{self.name}.pt'
    sd = self.state_dict()
    sd['G'] = self.G
    th.save(sd, path)
    print(path)

  def load(self, dir, device=None):
    path = dir / f'{self.name}.pt'
    sd = th.load(path, map_location=self.G.device)
    G = sd.pop('G')
    self.load_state_dict(sd)
    print(f'LOADED {path}')
