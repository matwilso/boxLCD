import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import nest
from torch import optim
import numpy as np
from algo.base import Trainer
from torch import distributions
import utils
from torch.cuda import amp
from nets import models

def fvmap(f):
    """fakes vmap by just reshaping"""
    def _thunk(x):
        # TODO: add support for nest.mapping this over a dict of stuff.
        preshape = x.shape
        x = f(x.flatten(0, 1))
        return x.reshape(preshape[:2]+x.shape[1:])
    return _thunk

class Dyn(Trainer, nn.Module):
    def __init__(self, cfg, make_env):
        nn.Module.__init__(self)
        Trainer.__init__(self, cfg, make_env)
        if cfg.use_image:
            self.encoder = models.ConvEncoder(cfg).to(cfg.device)
            self.decoder = models.ConvDecoder(cfg).to(cfg.device)
        else:
            self.encoder = models.DenseEncoder(self.state_shape[0], cfg).to(cfg.device)
            self.decoder = models.DenseDecoder(self.state_shape[0], cfg).to(cfg.device)
        self.dynamics = models.RSSM(self.act_n, cfg).to(cfg.device)
        self.params = itertools.chain(self.encoder.parameters(), self.decoder.parameters(), self.dynamics.parameters())
        self.optimizer = optim.Adam(self.params, lr=self.cfg.dyn_lr)
        # TODO: try out just autoencoder first
        self.venc = fvmap(self.encoder)
        #self.scaler = amp.GradScaler()

    def image_summaries(self, data, embed, obs_pred):
        truth = data['image'][:5] + 0.5
        recon = obs_pred.mean
        recon = recon.reshape((50, 50)+recon.shape[1:])[:5]
        init, _ = self.dynamics.observe(embed[:5, :5], data['act'][:5, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self.dynamics.imagine(data['act'][:5, 5:], init)
        openl = self.decoder(self.dynamics.get_feat(prior).flatten(0,1)).mean
        openl = openl.reshape((5, 45) + openl.shape[1:])
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1) # burn in for 5 steps (recon), then predict forward
        error = (model - truth + 1) / 2
        openl = torch.cat([truth, model, error], 3)
        openl = openl.permute([1, 2, 3, 0, 4]).flatten(-2,-1)[None]
        self.writer.add_video('agent/openl', openl, self.t, fps=20)

    def state_summaries(self, data, embed, obs_pred):
        truth = np.zeros([50, 3, 64, 64])
        for i in range(50):
            truth[i] = self.tenv.env.visualize_obs(np.array(data['state'][0][i].detach().cpu())).transpose(2,0,1) / 255.0

        recon = obs_pred.mean
        recon = recon.reshape((50, 50)+recon.shape[1:])[0]
        init, _ = self.dynamics.observe(embed[:5, :5], data['act'][:5, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self.dynamics.imagine(data['act'][:5, 5:], init)
        openl = self.decoder(self.dynamics.get_feat(prior).flatten(0,1)).mean
        openl = openl.reshape((5, 45) + openl.shape[1:])

        model = np.zeros([50, 3, 64, 64])
        for i in range(5):
            model[i] = self.tenv.env.visualize_obs(np.array(recon[i].detach().cpu())).transpose(2,0,1) / 255.0
        for i in range(5, 50):
            model[i] = self.tenv.env.visualize_obs(np.array(openl[0,i-5].detach().cpu())).transpose(2,0,1) / 255.0

        error = (model - truth + 1) / 2
        openl = np.concatenate([truth, model, error], 3)[None]
        self.writer.add_video('agent/openl', openl, self.t, fps=20)


    def update(self, batch, log_extra=False):
        obs = batch['image'] if self.cfg.use_image else batch['state']

        logs = {}
        self.optimizer.zero_grad()
        embed = self.venc(obs)
        post, prior = self.dynamics.observe(embed, batch['act'])
        feat = self.dynamics.get_feat(post)
        obs_pred = self.decoder(feat.flatten(0,1))
        recon_loss = -obs_pred.log_prob(obs.flatten(0,1)).mean()
        prior_dist = self.dynamics.get_dist(prior)
        post_dist = self.dynamics.get_dist(post)
        div = distributions.kl_divergence(post_dist, prior_dist).mean() # TODO: figure out how to make this work for amp
        div = torch.max(div, torch.tensor(3.0).to(self.cfg.device))
        model_loss = self.cfg.kl_scale*div + recon_loss

        #self.scaler.scale(model_loss).backward()
        model_loss.backward()
        if log_extra:
            ct, norms = 0, 0
            for p in list(filter(lambda p: p.grad is not None, self.parameters())):
                norms += p.grad.data.norm(2)
                ct += 1
            logs['grad_norm'] = norms/ct
        self.optimizer.step()
        #self.scaler.step(self.optimizer)
        #self.scaler.update()

        logs['recon_loss'] = recon_loss
        if log_extra:
            logs['div'] = div
            logs['prior_ent'] = prior_dist.entropy()
            logs['post_ent'] = post_dist.entropy()
            #self.logger['model_grad_norm'] = post_dist.entropy()
            if self.cfg.use_image:
                self.image_summaries(batch, embed, obs_pred)
            else:
                self.state_summaries(batch, embed, obs_pred)
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
            if self.cfg.use_image:
                batch = nest.map_structure(lambda x: torch.tensor(x).to(self.cfg.device), batch)
                batch['image'] = (batch['image'].permute([0, 1, -1, 2, 3]) / 255.0) - 0.5
                #batch = nest.map_structure(lambda x: x.flatten(0,1), batch)
                batch['image'] = batch['image'][:,:-1]
            else:
                if 'image' in batch: batch.pop('image')
                batch = nest.map_structure(lambda x: torch.tensor(x).to(self.cfg.device), batch)
            batch['state'] = batch['state'][:,:-1]
            self.logger['bdt'] += [time.time() - bt]

            ut = time.time()
            #self.update(batch, log_extra=1)
            self.update(batch, log_extra=self.t%self.cfg.log_n==0)
            self.logger['udt'] += [time.time() - ut]
            if self.t % self.cfg.log_n == 0:
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