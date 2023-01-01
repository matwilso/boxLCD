import numpy as np
import torch
from jax.tree_util import tree_map
from torch import distributions as thd
from torch import nn

from research import utils
from research.nets.common import ResBlock3d

from ._base import Autoencoder, MultiStepAE, SingleStepAE

TEMPORAL_RES = 4


class MultiStepArbiter(MultiStepAE):
    def __init__(self, env, G):
        super().__init__(env, G)
        self.lcd_key = G.lcd_key
        self.z_size = 256
        state_n = env.observation_space.spaces['proprio'].shape[0]
        act_n = env.action_space.shape[0]
        self.encoder = Encoder(state_n, self.z_size, G)
        self.decoder = Decoder(act_n, state_n, self.z_size, G)
        self._init()

        def batch_proc(batch):
            new_batch = {}
            for key, val in batch.items():
                if 'lcd' in key:
                    new_batch[key] = batch[key][:, :, ::8]
                else:
                    new_batch[key] = batch[key][:, ::8]
            return new_batch

        self.batch_proc = batch_proc

        assert self.G.window == 32

    def _unprompted_eval(
        self, epoch=None, writer=None, metrics=None, batch=None, arbiter=None
    ):
        return {}

    def save(self, dir, batch):
        print("SAVED MODEL", dir)
        path = dir / f'{self.name}.pt'
        self.eval()
        self.encoder.eval()

        class TracedArbiter(nn.Module):
            def __init__(self, encoder, decoder) -> None:
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, batch):
                z = self.encoder(batch)
                dec = self.decoder(z)
                return z, dec[2]

        d = TracedArbiter(self.encoder, self.decoder)
        jit_enc = torch.jit.trace(d, self.batch_proc(batch))
        torch.jit.save(jit_enc, str(path))
        print(path)

    def loss(self, batch):
        z = self.encoder(batch)
        decoded = self.decoder.forward_dist(z)
        recon_losses = {}
        recon_losses['loss/recon_proprio'] = (
            -decoded['proprio'].log_prob(batch['proprio']).mean()
        )
        recon_losses[f'loss/recon_{self.G.lcd_key}'] = (
            -decoded[self.G.lcd_key].log_prob(batch[self.G.lcd_key]).mean()
        )
        recon_losses['loss/recon_action'] = (
            -decoded['action'].log_prob(batch['action'][:, :-1]).mean()
        )
        recon_loss = sum(recon_losses.values())
        metrics = {'loss/recon_total': recon_loss, **recon_losses}

        if self.G.entropy_bonus != 0.0:
            z_post = thd.Normal(z, 1)
            z_prior = thd.Normal(0, 1)
            kl_reg_loss = self.G.entropy_bonus * thd.kl_divergence(z_post, z_prior).mean()
            # full loss and metrics
            recon_loss += kl_reg_loss
            metrics['kl_loss'] = kl_reg_loss

        return recon_loss, metrics

    def encode(self, batch, flatten=None):
        return self.encoder(batch)

    def _decode(self, z):
        return self.decoder.forward_dist(z)


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
            nn.Flatten(-2),
            nn.Linear(TEMPORAL_RES * n, n),
        )
        nf = G.nfilter
        size = 4
        self.seq = nn.ModuleList(
            [
                nn.Conv3d(3, nf, 3, 2, padding=1),
                ResBlock3d(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv3d(nf, nf, 3, 2, padding=1),
                ResBlock3d(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv3d(nf, nf, (1, 3, 3), (1, 2, 2), padding=(0, 1, 1)),
                ResBlock3d(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Flatten(-4),
                nn.Linear(size * nf, out_size),
            ]
        )

    def forward(self, batch):
        state = batch['proprio']
        lcd = batch[self.lcd_key]
        emb = self.state_embed(state)
        x = lcd

        for layer in self.seq:
            if isinstance(layer, ResBlock3d):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, act_n, state_n, in_size, G):
        super().__init__()
        self.lcd_key = G.lcd_key
        # assert G.lcd_h == 16 and G.lcd_w == 32
        # W = G.lcd_w // 8
        nf = G.nfilter
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_size, nf, (1, 2, 2), 2),
            nn.ReLU(),
            nn.ConvTranspose3d(nf, nf, 4, 4, padding=0),
            nn.ReLU(),
            nn.Conv3d(nf, nf, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(nf, 3, (1, 4, 4), (1, 2, 2), padding=(0, 1, 1)),
        )
        n = G.hidden_size
        self.state_net = nn.Sequential(
            nn.Linear(in_size, n),
            nn.ReLU(),
            nn.Linear(n, TEMPORAL_RES * n),
            nn.ReLU(),
            utils.Reshape(-1, TEMPORAL_RES, n),
            nn.Linear(n, state_n),
        )
        self.act_net = nn.Sequential(
            nn.Linear(in_size, n),
            nn.ReLU(),
            nn.Linear(n, (TEMPORAL_RES - 1) * n),
            nn.ReLU(),
            utils.Reshape(-1, TEMPORAL_RES - 1, n),
            nn.Linear(n, act_n),
        )

    def forward_dist(self, x):
        lcd, proprio, act = self.forward(x)
        lcd_dist = thd.Normal(torch.tanh(lcd), 1)
        # lcd_dist = thd.Bernoulli(logits=lcd)
        state_dist = thd.Normal(proprio, 1)
        act_dist = thd.Normal(act, 1)
        return {self.lcd_key: lcd_dist, 'proprio': state_dist, 'action': act_dist}

    def forward(self, x):
        lcd = self.net(x[..., None, None, None])
        proprio = self.state_net(x)
        act = self.act_net(x)
        return lcd, proprio, act
