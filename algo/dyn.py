import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from algo.base import Trainer
from tensorflow import nest
from torch import distributions
import utils


def fvmap(f):
    """fakes vmap by just reshaping"""
    def _thunk(x):
        # TODO: add support for nest.mapping this over a dict of stuff.
        preshape = x.shape
        x = f(x.flatten(0, 1))
        return x.reshape(preshape[:2]+x.shape[1:])
    return _thunk

class RSSM(nn.Module):
    def __init__(self, act_n, cfg):
        super().__init__()
        self.cfg = cfg
        self.cell = nn.GRUCell(self.cfg.hidden, self.cfg.deter)
        self.obs1 = nn.Linear(1024+self.cfg.deter, self.cfg.hidden)
        self.obs2 = nn.Linear(self.cfg.hidden, 2*self.cfg.stoch)
        self.img1 = nn.Linear(self.cfg.stoch+act_n, self.cfg.hidden)
        self.img2 = nn.Linear(self.cfg.deter, self.cfg.hidden)
        self.img3 = nn.Linear(self.cfg.hidden, 2*self.cfg.stoch)

    def initial(self, bs):
        init = dict(mean=torch.zeros([bs, self.cfg.stoch]), std=torch.zeros([bs, self.cfg.stoch]), stoch=torch.zeros([bs, self.cfg.stoch]), deter=torch.zeros([bs, self.cfg.deter]))
        return nest.map_structure(lambda x: x.to(self.cfg.device), init)

    def observe(self, embed, action, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        embed = embed.permute([1, 0, 2])
        action = action.permute([1, 0, 2])
        post, prior = utils.static_scan(lambda prev, inputs: self.obs_step(prev[0], *inputs), (action, embed), (state, state))
        post = {k: v.permute([1, 0, 2]) for k, v in post.items()}
        prior = {k: v.permute([1, 0, 2]) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action.permute([1, 0, 2])
        prior = utils.static_scan(self.img_step, action, state)
        prior = {k: v.permute([1, 0, 2]) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        return torch.cat([state['stoch'], state['deter']], -1)

    def get_dist(self, state):
        return distributions.MultivariateNormal(state['mean'], scale_tril=torch.diag_embed(state['std']))

    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior['deter'], embed], -1)
        x = self.obs1(x)
        x = F.relu(x)
        x = self.obs2(x)
        mean, std = torch.split(x, self.cfg.stoch, -1)
        std = F.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
        return post, prior

    def img_step(self, prev_state, prev_action):
        x = torch.cat([prev_state['stoch'], prev_action], -1)
        x = self.img1(x)
        x = F.relu(x)
        deter = self.cell(x, prev_state['deter'])
        x = self.img2(deter)
        x = F.relu(x)
        x = self.img3(x)
        mean, std = torch.split(x, self.cfg.stoch, -1)
        std = F.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
        return prior

class ConvEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._depth = 32
        self.c1 = nn.Conv2d(3, self._depth, kernel_size=4, stride=2)
        self.c2 = nn.Conv2d(self._depth, 2*self._depth,
                            kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(2*self._depth, 4*self._depth,
                            kernel_size=4, stride=2)
        self.c4 = nn.Conv2d(4*self._depth, 8*self._depth,
                            kernel_size=4, stride=2)

    def forward(self, image):
        x = self.c1(image)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.c4(x)
        return x.flatten(1, -1)


class ConvDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._depth = 32
        self._shape = (64, 64, 3)
        self.l1 = nn.Linear(self.cfg.deter+self.cfg.stoch, 32*self._depth)
        self.d1 = nn.ConvTranspose2d(
            32*self._depth, 4*self._depth, 5, stride=2)
        self.d2 = nn.ConvTranspose2d(4*self._depth, 2*self._depth, 5, stride=2)
        self.d3 = nn.ConvTranspose2d(2*self._depth, 1*self._depth, 6, stride=2)
        self.d4 = nn.ConvTranspose2d(1*self._depth, 3, 6, stride=2)

    def forward(self, features):
        x = self.l1(features)
        x = x[..., None, None]
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.relu(x)
        x = self.d3(x)
        x = F.relu(x)
        x = self.d4(x)
        #return distributions.Independent(distributions.Bernoulli(logits=x), len(self._shape))
        return distributions.Independent(distributions.Normal(x,1), len(self._shape))


class Dyn(Trainer, nn.Module):
    def __init__(self, cfg, make_env):
        nn.Module.__init__(self)
        Trainer.__init__(self, cfg, make_env)
        self.encoder = ConvEncoder(cfg).to(cfg.device)
        self.decoder = ConvDecoder(cfg).to(cfg.device)
        self.dynamics = RSSM(self.act_n, cfg).to(cfg.device)
        self.params = itertools.chain(self.encoder.parameters(), self.decoder.parameters(), self.dynamics.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.cfg.dyn_lr)
        # TODO: try out just autoencoder first
        self.venc = fvmap(self.encoder)

    def image_summaries(self, data, embed, image_pred):
        truth = data['image'][:4] + 0.5
        recon = image_pred.mean
        recon = recon.reshape((50, 50)+recon.shape[1:])[:4]
        init, _ = self.dynamics.observe(embed[:4, :5], data['act'][:4, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self.dynamics.imagine(data['act'][:4, 5:], init)
        openl = self.decoder(self.dynamics.get_feat(prior).flatten(0,1)).mean
        openl = openl.reshape((4, 45) + openl.shape[1:])
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        openl = torch.cat([truth, model, error], 3)
        openl = openl.permute([1, 2, 3, 4, 0]).flatten(-2,-1)[None]
        self.writer.add_video('agent/openl', openl, self.t, fps=20)

    def update(self, batch, log_extra=False):
        logs = {}
        self.optimizer.zero_grad()
        embed = self.venc(batch['image'])
        post, prior = self.dynamics.observe(embed, batch['act'])
        feat = self.dynamics.get_feat(post)
        image_pred = self.decoder(feat.flatten(0,1))
        recon_loss = -image_pred.log_prob(batch['image'].flatten(0,1)).mean()
        prior_dist = self.dynamics.get_dist(prior)
        post_dist = self.dynamics.get_dist(post)
        div = distributions.kl_divergence(post_dist, prior_dist).mean()
        div = torch.max(div, torch.tensor(3.0).to(self.cfg.device))
        model_loss = self.cfg.kl_scale*div + recon_loss
        model_loss.backward()
        if log_extra:
            ct, norms = 0, 0
            for p in list(filter(lambda p: p.grad is not None, self.parameters())):
                norms += p.grad.data.norm(2)
                ct += 1
            logs['grad_norm'] = norms/ct
        self.optimizer.step()

        logs['recon_loss'] = recon_loss
        if log_extra:
            logs['div'] = div
            logs['prior_ent'] = prior_dist.entropy()
            logs['post_ent'] = post_dist.entropy()
            #self.logger['model_grad_norm'] = post_dist.entropy()
            self.image_summaries(batch, embed, image_pred)

        for key in logs:
            self.logger[key] += [logs[key].mean().detach().cpu()]
        return model_loss

        #for i in range(len(batch['act'][0])):
        #    sb = nest.map_structure(lambda x: x[i], batch)
        #    self.optimizer.zero_grad()
        #    embed = self.encoder(sb)
        #    decode = self.decoder(embed)
        #    loss = -decode.log_prob(sb['image']).mean()
        #    loss.backward()
        #    self.optimizer.step()
        #    self.logger['Loss'] += [loss.detach().cpu()]
        #    info = {}
        #    info['recon'] = (decode.mean[:6] + 0.5).detach().cpu()
        #return info

    def run(self):
        self.refresh_dataset()
        start_time = time.time()
        epoch_time = time.time()
        for self.t in itertools.count(1):
            bt = time.time()
            batch = next(self.data_iter)
            batch = nest.map_structure(
                lambda x: torch.tensor(x).to(self.cfg.device), batch)
            batch['image'] = (batch['image'].permute([0, 1, -1, 2, 3]) / 255.0) - 0.5
            #batch = nest.map_structure(lambda x: x.flatten(0,1), batch)
            batch['image'] = batch['image'][:,:-1]
            batch['state'] = batch['state'][:,:-1]
            self.logger['bdt'] += [time.time() - bt]

            ut = time.time()
            self.update(batch, log_extra=1)
            #self.update(batch, log_extra=self.t%100==0)
            self.logger['udt'] += [time.time() - ut]
            if self.t % 1 == 0:
                print('='*30)
                print('t', self.t)
                for key in self.logger:
                    x = np.mean(self.logger[key])
                    self.writer.add_scalar(key, x, self.t)
                    print(key, x)
                self.writer.flush()
                print('dt', time.time()-epoch_time)
                print('total time', time.time()-start_time)
                print(self.logpath)
                print(self.cfg.full_cmd)
                print('='*30)
                epoch_time = time.time()
