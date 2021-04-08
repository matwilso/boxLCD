import torch as th
from torch import nn
from torch.optim import Adam

# TODO: something to autoregister all nets
class Net(nn.Module):
  def __init__(self, env, C):
    super().__init__()
    self.C = C
    self.env = env
    self.imsize = self.C.lcd_h * self.C.lcd_w
    self.act_n = env.action_space.shape[0]

  def _init(self):
    self.optimizer = Adam(self.parameters(), lr=C.lr)
    self.to(self.C.device)

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
    th.save(self.state_dict(), path)
    print(path)

  def load(self, path):
    path = path / f'{self.name}.pt'
    self.load_state_dict(th.load(path))
    print(f'LOADED {path}')
