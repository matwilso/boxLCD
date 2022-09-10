from re import I

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import distributions as thd
from torch import nn

from research import utils
from research.nets.common import ResBlock

from ._base import Autoencoder, SingleStepAE


class VAE(SingleStepAE):
    def __init__(self, env, G):
        super().__init__(env, G)
        self.z_size = 128
        state_n = env.observation_space.spaces['proprio'].shape[0]
        self.encoder = Encoder(state_n, self.z_size, G)
        self.decoder = Decoder(state_n, self.z_size, G)
        self._init()

    def sample_z(self, n):
        z = th.randn(n, self.G.z_size).to(self.G.device)
        return z

    def loss(self, batch):
        z_post = self.encoder(batch)
        decoded = self.decoder(z_post.rsample())
        # recon
        recon_losses = {}
        recon_losses['loss/recon_proprio'] = (
            -decoded['proprio'].log_prob(batch['proprio']).mean()
        )
        recon_losses['loss/recon_lcd'] = (
            -decoded['lcd'].log_prob(batch['lcd'][:, None]).mean()
        )
        recon_loss = sum(recon_losses.values())
        # kl div constraint
        z_prior = thd.Normal(0, 1)
        kl_loss = thd.kl_divergence(z_post, z_prior).mean(-1)
        # full loss and metrics
        loss = (recon_loss + self.G.beta * kl_loss).mean()
        metrics = {
            'loss/vae_loss': loss,
            'loss/kl': kl_loss.mean(),
            'loss/recon_total': recon_loss,
            **recon_losses,
        }
        return loss, metrics

    def encode(self, batch, flatten=None, noise=False):
        if noise:
            z_post = self.encoder(batch).sample()
        else:
            z_post = self.encoder(batch).mean
        return z_post

    def _decode(self, z):
        return self.decoder(z)


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
                nn.Linear(size * nf, 2 * out_size),
            ]
        )

    def get_dist(self, x):
        mu, log_std = x.chunk(2, -1)
        std = F.softplus(log_std) + 1e-4
        return thd.Normal(mu, std)

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
        return self.get_dist(x)


class Decoder(nn.Module):
    def __init__(self, state_n, in_size, G):
        super().__init__()
        assert G.lcd_h == 16, G.lcd_w == 32
        nf = G.nfilter
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_size, nf, (2, 4), 2),
            nn.ReLU(),
            nn.ConvTranspose2d(nf, nf, 4, 4, padding=0),
            nn.ReLU(),
            nn.Conv2d(nf, nf, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(nf, 1, 4, 2, padding=1),
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
        lcd_dist = thd.Bernoulli(logits=self.net(x[..., None, None]))
        state_dist = thd.Normal(self.state_net(x), 1)
        return {'lcd': lcd_dist, 'proprio': state_dist}
