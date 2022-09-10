import torch
from torch import distributions as thd
from torch import nn

from research import utils
from research.nets.common import ResBlock

from ._base import Autoencoder, SingleStepAE
from .vae import Encoder


class ObjectLocalizer(SingleStepAE):
    def __init__(self, env, G):
        super().__init__(env, G)
        self.z_size = 128
        state_n = env.observation_space.spaces['proprio'].shape[0]
        self.encoder = Encoder(state_n, 2, G)
        self.keys = utils.filtlist(self.env.obs_keys, 'object.*(x|y):p')
        self.idxs = [self.env.obs_keys.index(x) for x in self.keys]
        self._init()

    def evaluate(self, epoch, writer, batch, arbiter=None):
        return {}

    def save(self, dir, batch):
        print("SAVED MODEL", dir)
        path = dir / f'{self.name}.pt'
        jit_enc = torch.jit.trace(self.encoder, self.batch_proc(batch))
        torch.jit.save(jit_enc, str(path))
        print(path)

    def loss(self, batch):
        z, z_std = self.encoder(batch)
        norm = thd.Normal(z, z_std)
        loss = -norm.log_prob(batch['full_state'][..., self.idxs]).mean()
        # loss = ((batch['full_state'][...,self.idxs] - z)**2).mean()
        return loss, {'loss': loss}


class Encoder(nn.Module):
    def __init__(self, state_n, out_size, G):
        super().__init__()
        n = G.hidden_size
        self.state_embed = nn.Sequential(
            nn.Linear(state_n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
        )
        nf = G.nfilter
        size = (G.lcd_h * G.lcd_w) // 64
        self.seq = nn.ModuleList(
            [
                nn.Conv2d(1, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv2d(nf, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv2d(nf, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Flatten(-3),
                nn.Linear(size * nf, G.hidden_size),
                nn.ReLU(),
                nn.Linear(G.hidden_size, 2 * out_size),
            ]
        )

    def forward(self, batch):
        state = batch['proprio']
        lcd = batch['lcd']
        emb = self.state_embed(state)
        x = lcd[:, None]
        for layer in self.seq:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        mean, log_std = x.chunk(2, -1)
        return mean, torch.exp(log_std)
