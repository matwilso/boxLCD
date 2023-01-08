import ignite
import torch
from torch import nn
from torch.optim import Adam

from research.define_config import env_fn
from torch.cuda import amp


class Net(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G
        self.ssim = ignite.metrics.SSIM(1.0, device=self.G.device)
        self.psnr = ignite.metrics.PSNR(1.0, device=self.G.device)
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.arbiter = None
        self.name = self.__class__.__name__
        self.scaler = amp.GradScaler()

    @classmethod
    def cname(cls):
        breakpoint()
        cls.name = cls.__class__.__name__

    @classmethod
    def from_disk(cls, dir, device='cuda'):
        sd = torch.load(dir, map_location=device)
        G = sd.pop('G')
        env = env_fn(G)
        net = cls(env, G)
        net.load_state_dict(sd)
        net.to(device)
        return net

    def _init(self):
        self.optimizer = Adam(self.parameters(), lr=self.G.lr)

    def train_step_amp(self, batch, dry=False):
        self.optimizer.zero_grad()
        with amp.autocast():
            loss, metrics = self.loss(batch)
        if not dry:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            metrics['meta/loss_scale'] = torch.tensor(self.scaler.get_scale())
        return metrics

    def train_step(self, batch, dry=False):
        if self.G.amp:
            return self.train_step_amp(batch, dry)
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
        torch.save(sd, path)
        print(path)

    def load(self, dir, device=None):
        device = device or self.G.device
        path = dir / f'{self.name}.pt'
        sd = torch.load(path, map_location=self.G.device)
        G = sd.pop('G')
        self.load_state_dict(sd)
        print(f'LOADED {path}')
