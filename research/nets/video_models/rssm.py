from itertools import chain, count

import ignite
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jax.tree_util import tree_map, tree_multimap
from torch import distributions as thd
from torch import nn

from research import utils
from research.nets.common import (
    BinaryHead,
    ConvBinHead,
    ConvEmbed,
    GaussHead,
    MDNHead,
    MultiHead,
    ResBlock,
    TransformerBlock,
    aggregate,
)

from ._base import VideoModel


class RSSM(VideoModel):
    def __init__(self, env, G):
        super().__init__(env, G)
        self._stoch_size = 64
        self._deter_size = 256
        state_n = env.observation_space.spaces['proprio'].shape[0]
        self.embed_size = 256
        self.encoder = Encoder(state_n, self.embed_size, G)
        self.cell = nn.GRUCell(self.G.hidden_size, self._deter_size)
        self.decoder = Decoder(state_n, self._stoch_size + self._deter_size, G)
        self.obs_net = nn.Sequential(
            nn.Linear(self.embed_size + self._deter_size, self.G.hidden_size),
            nn.ReLU(),
            nn.Linear(self.G.hidden_size, 2 * self._stoch_size),
        )
        self.img1 = nn.Linear(
            self._stoch_size + env.action_space.shape[0], self.G.hidden_size
        )
        self.img_net = nn.Sequential(
            nn.Linear(self._deter_size, self.G.hidden_size),
            nn.ReLU(),
            nn.Linear(self.G.hidden_size, 2 * self._stoch_size),
        )
        self._init()

    def loss(self, batch):
        batch = rearrange
        flat_batch = tree_map(lambda x: x.flatten(0, 1), batch)
        import ipdb

        ipdb.set_trace()
        embed = self.encoder(flat_batch).unflatten(0, (*batch['lcd'].shape[:2],))
        action = batch['action'][:, :-1]
        embed = embed[:, 1:]
        post, prior = self.observe(embed, action)
        feat = self.get_feat(post)
        # reconstruction loss
        decoded = self.decoder(feat.flatten(0, 1))
        recon_losses = {}
        chop_flat = tree_map(lambda x: x[:, 1:].flatten(0, 1), batch)
        recon_losses['loss/recon_proprio'] = (
            -decoded['proprio'].log_prob(chop_flat['proprio']).mean()
        )
        recon_losses['loss/recon_lcd'] = (
            -decoded['lcd'].log_prob(chop_flat['lcd']).mean()
        )
        recon_loss = sum(recon_losses.values())

        # variational inference loss.
        # make it so that we can reconstruct obs with only seeing action.
        prior_dist = self.get_dist(prior)
        post_dist = self.get_dist(post)
        # TODO: assert they are same size
        div = thd.kl_divergence(post_dist, prior_dist)
        div = torch.max(div, self.G.free_nats * torch.ones_like(div)).mean()
        div_loss = self.G.kl_scale * div
        loss = recon_loss + div_loss
        return loss, {
            'div_loss': div_loss,
            'loss/total': loss,
            **recon_losses,
            'loss/recon_total': recon_loss,
        }

    def initial(self, batch_size):
        state = dict(
            mean=torch.zeros([batch_size, self._stoch_size]),
            std=torch.zeros([batch_size, self._stoch_size]),
            stoch=torch.zeros([batch_size, self._stoch_size]),
            deter=torch.zeros([batch_size, self._deter_size]),
        )
        return tree_map(lambda x: x.to(self.G.device), state)

    def observe(self, embed, action, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        posts, priors = [], []
        for i in range(action.shape[1]):
            post, prior = self.obs_step(state, action[:, i], embed[:, i])
            posts += [post]
            priors += [prior]
            state = post
        posts = tree_multimap(lambda x, *y: torch.stack([x, *y], 1), posts[0], *posts[1:])
        priors = tree_multimap(
            lambda x, *y: torch.stack([x, *y], 1), priors[0], *priors[1:]
        )
        return posts, priors

    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior['deter'], embed], -1)
        x = self.obs_net(x)
        mean, std = torch.chunk(x, 2, -1)
        std = F.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).rsample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
        return post, prior

    def img_step(self, prev_state, prev_action):
        x = torch.cat([prev_state['stoch'], prev_action], -1)
        x = F.relu(self.img1(x))
        x = deter = self.cell(x, prev_state['deter'])
        x = self.img_net(x)
        mean, std = torch.chunk(x, 2, -1)
        std = F.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).rsample()
        prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
        return prior

    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        priors = []
        for i in range(action.shape[1]):
            prior = self.img_step(state, action[:, i])
            priors += [prior]
            state = prior
        priors = tree_multimap(
            lambda x, *y: torch.stack([x, *y], 1), priors[0], *priors[1:]
        )
        return priors

    def sample(self, n, action=None, prompts=None, prompt_n=10):
        with torch.no_grad():
            if action is not None:
                n = action.shape[0]
            else:
                action = (torch.rand(n, self.G.window, self.act_n) * 2 - 1).to(
                    self.G.device
                )
            if prompts is None:
                prior = self.imagine(action)
                feat = self.get_feat(prior)
                decoded = self.decoder(feat.flatten(0, 1))
                gen = {
                    'lcd': (1.0 * (decoded['lcd'].probs > 0.5)),
                    'proprio': decoded['proprio'].mean,
                }
                gen['lcd'] = gen['lcd'].reshape(n, -1, 1, self.G.lcd_h, self.G.lcd_w)
                gen['proprio'] = gen['proprio'].reshape(
                    n, -1, self.env.observation_space['proprio'].shape[0]
                )
            else:
                batch = tree_map(lambda x: x[:, :prompt_n], prompts)
                flat_batch = tree_map(lambda x: x.flatten(0, 1), batch)
                embed = self.encoder(flat_batch).unflatten(
                    0, (*batch['lcd'].shape[:2],)
                )
                action = torch.cat([th.zeros_like(action)[:, :1], action[:, :-1]], 1)
                post, prior = self.observe(embed, action[:, :prompt_n])
                prior = self.imagine(
                    action[:, prompt_n:], state=tree_map(lambda x: x[:, -1], post)
                )
                feat = self.get_feat(prior)
                decoded = self.decoder(feat.flatten(0, 1))
                gen = {
                    'lcd': (1.0 * (decoded['lcd'].probs > 0.5)),
                    'proprio': decoded['proprio'].mean,
                }
                gen['lcd'] = gen['lcd'].reshape(n, -1, 1, self.G.lcd_h, self.G.lcd_w)
                gen['proprio'] = gen['proprio'].reshape(
                    n, -1, self.env.observation_space['proprio'].shape[0]
                )
                prompts['lcd'] = prompts['lcd'][:, :, None]
                gen = tree_multimap(
                    lambda x, y: torch.cat([x[:, :prompt_n], y], 1),
                    utils.subdict(prompts, ['lcd', 'proprio']),
                    gen,
                )
                prompts['lcd'] = prompts['lcd'][:, :, 0]
        return gen

    def get_feat(self, state):
        return torch.cat([state['stoch'], state['deter']], -1)

    def get_dist(self, state):
        return thd.Normal(state['mean'], state['std'])
        # return thd.MultivariateNormal(state['mean'], scale_tril=torch.diag_embed(state['std']))


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
                nn.Conv2d(3, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv2d(nf, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Conv2d(nf, nf, 3, 2, padding=1),
                ResBlock(nf, emb_channels=G.hidden_size, group_size=4),
                nn.Flatten(-3),
                nn.Linear(size * nf, out_size),
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
        return x


class Decoder(nn.Module):
    def __init__(self, state_n, in_size, G):
        super().__init__()
        nf = G.nfilter
        H = 2
        W = int(2 * G.wh_ratio)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_size, nf, (H, W), 2),
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
        lcd_dist = thd.Bernoulli(logits=self.net(x[..., None, None]))
        state_dist = thd.Normal(self.state_net(x), 1)
        return {'lcd': lcd_dist, 'proprio': state_dist}
