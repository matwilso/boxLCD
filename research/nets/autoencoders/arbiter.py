import torch
from torch import distributions as thd
from torch import nn

from research import utils
from research.nets.common import ResBlock

from ._base import Autoencoder, SingleStepAE


class ArbiterAE(SingleStepAE):
    def __init__(self, env, G):
        super().__init__(env, G)
        self.z_size = 128
        state_n = env.observation_space.spaces['proprio'].shape[0]
        self.res = G.dst_resolution
        self.encoder = Encoder(state_n, self.z_size, G)
        self.decoder = Decoder(state_n, self.z_size, G)
        self._init()

    def _unprompted_eval(
        self, epoch=None, writer=None, metrics=None, batch=None, arbiter=None
    ):
        return {}

    def save(self, dir, batch):
        print("SAVED MODEL", dir)
        path = dir / f'{self.name}.pt'
        jit_enc = torch.jit.trace(self.encoder, self.batch_proc(batch))
        torch.jit.save(jit_enc, str(path))
        print(path)

    def loss(self, batch):
        z = self.encoder(batch)
        decoded = self.decoder(z)
        recon_losses = {}
        recon_losses['loss/recon_proprio'] = (
            -decoded['proprio'].log_prob(batch['proprio']).mean()
        )
        recon_losses['loss/recon_lcd'] = -decoded[self.G.lcd_key].log_prob(batch[self.G.lcd_key]).mean()
        recon_loss = sum(recon_losses.values())
        metrics = {'loss/recon_total': recon_loss, **recon_losses}
        return recon_loss, metrics

    def encode(self, batch, flatten=None, noise=None):
        return self.encoder(batch)

    def _decode(self, z):
        return self.decoder(z)


class Encoder(nn.Module):
    def __init__(self, state_n, out_size, G):
        super().__init__()
        self.lcd_key = G.lcd_key
        n = G.hidden_size
        self.state_embed = nn.Sequential(
            nn.Linear(state_n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
        )
        nf = G.nfilter
        size = 4
        if G.resolution == 16:
            downs = (3, 2)
        elif G.resolution == 32:
            downs = (5, 4)
        elif G.resolution == 64:
            downs = (10, 8)

        self.seq = nn.ModuleList(
            [
                nn.Conv2d(3, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv2d(nf, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv2d(nf, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv2d(nf, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv2d(nf, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Flatten(-3),
                nn.Linear(size * nf, out_size),
            ]
        )

    def forward(self, batch):
        state = batch['proprio']
        x = batch[self.lcd_key]
        emb = self.state_embed(state)
        for layer in self.seq:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, state_n, in_size, G):
        super().__init__()
        self.lcd_key = G.lcd_key
        nf = G.nfilter
        up_res = G.resolution // 8
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_size, nf, (up_res, up_res), 2),
            nn.ReLU(),
            nn.ConvTranspose2d(nf, nf, 4, 4, padding=0),
            nn.ReLU(),
            nn.Conv2d(nf, nf, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(nf, 3, 4, 2, padding=1),
        )
        n = G.hidden_size
        self.state_net = nn.Sequential(
            nn.Linear(in_size, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, state_n),
        )
        nf = G.nfilter

    def forward(self, x):
        lcd_dist = thd.Normal(torch.tanh(self.net(x[..., None, None])), 1)
        state_dist = thd.Normal(self.state_net(x), 1)
        return {self.lcd_key: lcd_dist, 'proprio': state_dist}
