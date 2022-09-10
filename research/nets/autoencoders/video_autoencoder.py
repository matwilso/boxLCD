import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from jax.tree_util import tree_map
from torch import distributions as thd
from torch import nn

from research import utils
from research.nets.common import ResBlock3d

from ._base import Autoencoder, MultiStepAE, SingleStepAE


class VideoAutoencoder(MultiStepAE):
    def __init__(self, env, G):
        super().__init__(env, G)
        state_n = env.observation_space.spaces['proprio'].shape[0]
        act_n = env.action_space.shape[0]
        self.encoder = Encoder(state_n, G)
        self.decoder = Decoder(act_n, state_n, G)
        self._init()

    def _unprompted_eval(
        self, epoch=None, writer=None, metrics=None, batch=None, arbiter=None
    ):
        return {}

    def save(self, dir, batch):
        print("SAVED MODEL", dir)
        path = dir / f'{self.name}.pt'
        self.eval()
        self.encoder.eval()

        class TracedEncoder(nn.Module):
            def __init__(self, encoder) -> None:
                super().__init__()
                self.encoder = encoder

            def forward(self, batch):
                z = self.encoder(batch)
                return z

        class TracedDecoder(nn.Module):
            def __init__(self, decoder) -> None:
                super().__init__()
                self.decoder = decoder

            def forward(self, z):
                dec = self.decoder.forward(z)
                return dec

        enc = TracedEncoder(self.encoder)
        batch = self.batch_proc(batch)
        jit_enc = torch.jit.trace(enc, batch, strict=False)
        torch.jit.save(jit_enc, str(path.with_name('Encoder.pt')))

        z = jit_enc(batch)
        dec = TracedDecoder(self.decoder)
        jit_dec = torch.jit.trace(dec, z, strict=False)
        torch.jit.save(jit_dec, str(path.with_name('Decoder.pt')))
        print(path)

    def loss(self, batch):
        z = self.encoder(batch)
        decoded = self.decoder.forward_dist(z)
        recon_losses = {}
        recon_losses['loss/recon_proprio'] = (
            -decoded['proprio'].log_prob(batch['proprio']).mean()
        )
        recon_losses['loss/recon_lcd'] = -decoded['lcd'].log_prob(batch['lcd']).mean()
        recon_losses['loss/recon_action'] = (
            -decoded['action'].log_prob(batch['action'][:, :-1]).mean()
        )
        recon_loss = sum(recon_losses.values())
        metrics = {'loss/recon_total': recon_loss, **recon_losses}

        metrics['kl/std'] = z.std()
        if self.G.entropy_bonus != 0.0:
            z_post = thd.Normal(z, 1)
            z_prior = thd.Normal(0, 1)
            kl_reg_loss = thd.kl_divergence(z_post, z_prior).mean()
            metrics['kl/loss'] = kl_reg_loss
            recon_loss += self.G.entropy_bonus * kl_reg_loss

        return recon_loss, metrics

    def encode(self, batch, flatten=None):
        return self.encoder(batch)

    def _decode(self, z):
        return self.decoder.forward_dist(z)


class Encoder(nn.Module):
    def __init__(self, state_n, G):
        super().__init__()
        n = G.hidden_size
        self.state_embed = nn.Sequential(
            nn.Linear(state_n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Flatten(-2),
            nn.Linear(G.window * n, n),
        )
        nf = G.nfilter
        in_channels = 3  # or 1 depending on lcd mode = '1'
        if G.window == 4:
            stride = 2
        else:
            stride = 1

        self.seq = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels,
                    nf,
                    kernel_size=(3, 3, 3),
                    stride=(stride, 2, 2),
                    padding=1,
                ),
                ResBlock3d(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv3d(
                    nf, nf, kernel_size=(3, 3, 3), stride=(stride, 2, 2), padding=1
                ),
                ResBlock3d(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv3d(nf, nf, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
                ResBlock3d(nf, emb_channels=G.hidden_size, group_size=4),
            ]
        )

    def forward(self, batch):
        state = batch['proprio']
        lcd = batch['lcd']
        emb = self.state_embed(state)
        x = lcd
        for layer in self.seq:
            if isinstance(layer, ResBlock3d):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, act_n, state_n, G):
        super().__init__()
        # assert G.lcd_h == 16 and G.lcd_w == 32
        if G.lcd_w == 32:
            W = 4
        elif G.lcd_w == 24:
            W = 3
        elif G.lcd_w == 16:
            W = 2

        if G.window == 4:
            stride = 2
        else:
            stride = 1

        nf = G.nfilter
        self.net = nn.Sequential(
            nn.ConvTranspose3d(
                nf, nf, kernel_size=(stride, 2, W), stride=(stride, 2, 2)
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                nf, nf, kernel_size=(stride, 2, 2), stride=(stride, 2, 2)
            ),
            nn.ReLU(),
            nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(nf, 3, kernel_size=1, stride=1, padding=0),
        )

        self.flat_proc = nn.Sequential(
            Rearrange('b c d h w -> b d c h w'),
            nn.Conv3d(1, G.window, kernel_size=1, stride=1),
            Rearrange('b t c h w -> b t (c h w)'),
        )

        n = G.nfilter
        self.state_net = nn.Sequential(
            nn.Linear(n * 16, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, state_n),
        )
        self.act_net = nn.Sequential(
            nn.Linear(n * 16, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, act_n),
        )

    @staticmethod
    def dist(dec):
        lcd_dist = thd.Bernoulli(logits=dec['lcd'], validate_args=False)
        state_dist = thd.Normal(dec['proprio'], 1)
        act_dist = thd.Normal(dec['action'], 1)
        return {'lcd': lcd_dist, 'proprio': state_dist, 'action': act_dist}

    def forward_dist(self, x):
        return self.dist(self.forward(x))

    def forward(self, x):
        lcd = self.net(x)
        flatx = self.flat_proc(x)
        proprio = self.state_net(flatx)
        # TODO: should it be 1: or :-1?
        act = self.act_net(flatx[:, 1:])
        return {'lcd': lcd, 'proprio': proprio, 'action': act}
