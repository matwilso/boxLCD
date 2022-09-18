import torch
import torch.nn.functional as F
from einops import rearrange
from jax.tree_util import tree_map, tree_multimap
from torch import distributions as thd
from torch import nn

from research import utils
from research.nets.common import ResBlock

from ._base import VideoModel

from einops import parse_shape, rearrange


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
        bs = batch['lcd'].shape[0]

        # TODO: write utility to pack and unpack different dims
        def flatten(key, val, trim=None):
            if key == 'lcd':
                return rearrange(val[:,:,trim:], 'b c t h w -> (b t) c h w')
            else:
                return rearrange(val[:,trim:], 'b t x -> (b t) x')

        def unflatten(key, val):
            if key == 'lcd':
                return rearrange(val, '(b t) c h w -> b c t h w', b=bs)
            else:
                return rearrange(val, '(b t) x -> b t x', b=bs)


        batch['lcd'] = batch['lcd']
        flat_batch = {key: flatten(key, val) for key, val in batch.items()}
        embed = unflatten('z', self.encoder(flat_batch))
        action = batch['action'][:, :-1]
        embed = embed[:, 1:]
        post, prior = self.observe(embed, action)
        feat = self.get_feat(post)
        # reconstruction loss
        decoded = self.decoder(feat.flatten(0, 1))
        batch_with_first_time_gone = {key: flatten(key, val, trim=1) for key, val in batch.items()}
        recon_losses = {}
        recon_losses['loss/recon_proprio'] = (
            -decoded['proprio'].log_prob(batch_with_first_time_gone['proprio']).mean()
        )
        recon_losses['loss/recon_lcd'] = (
            -decoded['lcd'].log_prob(batch_with_first_time_gone['lcd']).mean()
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
        posts = tree_multimap(
            lambda x, *y: torch.stack([x, *y], 1), posts[0], *posts[1:]
        )
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
        bs = n

        def trim_dim(key, val, trim_start=None, trim_end=None):
            if key == 'lcd':
                return val[:, :, trim_start:trim_end]
            else:
                return val[:, trim_start:trim_end]

        def flatten(key, val, trim=None, trim_end=None):
            if key == 'lcd':
                return rearrange(val[:,:,trim:trim_end], 'b c t h w -> (b t) c h w')
            else:
                return rearrange(val[:,trim:trim_end], 'b t x -> (b t) x')

        def unflatten(key, val):
            if key == 'lcd':
                return rearrange(val, '(b t) c h w -> b c t h w', b=bs)
            else:
                return rearrange(val, '(b t) x -> b t x', b=bs)


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
                    'lcd': decoded['lcd'].probs,
                    'proprio': decoded['proprio'].mean,
                }
                gen['lcd'] = gen['lcd'].reshape(n, -1, 1, self.G.lcd_h, self.G.lcd_w)
                gen['proprio'] = gen['proprio'].reshape(
                    n, -1, self.env.observation_space['proprio'].shape[0]
                )
            else:
                batch = {key: trim_dim(key, val, None, prompt_n) for key, val in prompts.items()}
                flat_batch = {key: flatten(key, val) for key, val in batch.items()}
                embed = unflatten('z', self.encoder(flat_batch))
                action = torch.cat([torch.zeros_like(action)[:, :1], action[:, :-1]], 1)
                post, prior = self.observe(embed, action[:, :prompt_n])
                prior = self.imagine(
                    action[:, prompt_n:], state=tree_map(lambda x: x[:, -1], post)
                )
                feat = self.get_feat(prior)
                decoded = self.decoder(feat.flatten(0, 1))
                gen = {
                    'lcd': decoded['lcd'].probs,
                    'proprio': decoded['proprio'].mean,
                }
                gen = {key: unflatten(key, val) for key, val in gen.items()}
                gen = tree_multimap(
                    lambda x, y: torch.cat([x, y], dim=2 if x.ndim == 5 else 1),
                    utils.subdict(batch, ['lcd', 'proprio']),
                    utils.subdict(gen, ['lcd', 'proprio']),
                )
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
        x = lcd
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
        lcd_dist = thd.Bernoulli(logits=self.net(x[..., None, None]), validate_args=False)
        state_dist = thd.Normal(self.state_net(x), 1)
        return {'lcd': lcd_dist, 'proprio': state_dist}
