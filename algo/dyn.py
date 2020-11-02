import time
import itertools
import torch
from jax.tree_util import tree_multimap
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import nest
from torch import optim
import numpy as np
from algo.trainer import Trainer
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
        self.latent_fwds = models.LatentFwds(self.act_n, cfg).to(cfg.device)

        # dreamer stuff
        self.value = models.DenseDecoder(1, cfg).to(cfg.device)
        self.actor = models.ActionDecoder(self.act_n, cfg).to(cfg.device)
        self.model_params = itertools.chain(self.encoder.parameters(), self.decoder.parameters(), self.dynamics.parameters())
        self.model_optimizer = optim.Adam(self.model_params, lr=self.cfg.dyn_lr)
        self.lds_optimizer = optim.Adam(self.latent_fwds.parameters(), lr=self.cfg.dyn_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.pi_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.cfg.vf_lr)

        # TODO: try out just autoencoder first
        self.venc = fvmap(self.encoder)
        #params = list(self.encoder.parameters())
        #params = nest.map_structure(lambda x: x[None].repeat(3, *([1]*len(x.shape))), params)
        #self.scaler = amp.GradScaler()
        #self.encoder._parameters = params

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
        recon = recon.reshape((self.cfg.bs, 50)+recon.shape[1:])[0]
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

    def imagine_ahead(self, post):
        start = {k: v.flatten(0,1) for k, v in post.items()}
        actions = torch.zeros([50*50, self.act_n]).to(self.cfg.device)
        # TODO: make policy autoregressive so we get better propagation.
        def func(prev, _):
            prev = prev[0]
            poo = torch.tanh(self.actor(self.dynamics.get_feat(prev)).rsample())
            return self.dynamics.img_step(prev, poo), poo
        outs = utils.static_scan(func, torch.range(0, self.cfg.horizon).to(self.cfg.device), (start, actions))
        state, actions = outs
        imag_feat = self.dynamics.get_feat(state)
        return imag_feat, actions

    def update(self, batch, log_extra=False):
        obs = batch['image'] if self.cfg.use_image else batch['state']
        logs = {}
        
        # MODEL UPDATE
        self.model_optimizer.zero_grad()
        self.lds_optimizer.zero_grad()
        embed = self.venc(obs)
        post, prior = self.dynamics.observe(embed, batch['act'])
        feat = self.dynamics.get_feat(post)
        obs_pred = self.decoder(feat.flatten(0,1))
        recon_loss = -obs_pred.log_prob(obs.flatten(0,1)).mean()
        #recon_loss = -obs_pred.log_prob(obs.flatten(0,1)).mean()
        prior_dist = self.dynamics.get_dist(prior)
        post_dist = self.dynamics.get_dist(post)
        div = distributions.kl_divergence(post_dist, prior_dist).mean() # TODO: figure out how to make this work for amp
        div = torch.max(div, torch.tensor(self.cfg.free_nats).to(self.cfg.device))
        model_loss = self.cfg.kl_scale*div + recon_loss
        #self.scaler.scale(model_loss).backward()
        feat = self.dynamics.get_feat(post).detach()
        lds = self.latent_fwds(torch.cat([feat, batch['act']], -1))
        lds_loss = (0.5*(feat[:,1:].detach() - lds[:,:,:-1])**2).mean()
        model_loss += lds_loss
        model_loss.backward()
        if log_extra:
            ct, norms = 0, 0
            for p in list(filter(lambda p: p.grad is not None, self.parameters())):
                norms += p.grad.data.norm(2)
                ct += 1
            logs['grad_norm'] = norms/ct
        self.model_optimizer.step()
        self.lds_optimizer.step()
        #self.scaler.step(self.model_optimizer)
        #self.scaler.update()
        # TODO: freeze model weights, but 
        # TODO: check gradients getting propped right.
        # ACTOR UPDATE
        self.actor_optimizer.zero_grad()
        post = {k: v.detach() for k,v in post.items()}
        imag_feat, actions = self.imagine_ahead(post)
        lds = self.latent_fwds(torch.cat([imag_feat, actions], -1))
        reward = lds.var(0).mean(-1).detach()
        pcont = self.cfg.gamma * torch.ones_like(reward)
        value = self.value(imag_feat).mean[...,0]
        returns = utils.lambda_return(reward[:-1], value[:-1], pcont[:-1], bootstrap=value[-1], lambda_=self.cfg.lam, axis=0)
        discount = torch.cumprod(torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0).detach()
        actor_loss = -(discount * returns).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # VALUE UPDATE
        self.value_optimizer.zero_grad()
        value_pred = self.value(imag_feat.detach())
        target = returns.detach()
        logp = value_pred.log_prob(torch.cat([target, torch.ones((1,)+target.shape[1:]).to(self.cfg.device)])[...,None])[:-1]
        value_loss = -(discount * logp).mean()
        self.value_optimizer.step()

        logs['recon_loss'] = recon_loss
        logs['div'] = div
        logs['model_loss'] = model_loss
        logs['value_loss'] = value_loss
        logs['actor_loss'] = actor_loss
        if log_extra:
            logs['prior_ent'] = prior_dist.entropy()
            logs['post_ent'] = post_dist.entropy()
            #self.logger['model_grad_norm'] = post_dist.entropy()
            lt = time.time()
            if self.cfg.use_image:
                self.image_summaries(batch, embed, obs_pred)
            else:
                self.state_summaries(batch, embed, obs_pred)
            self.logger['dt/summary'] += [time.time()-lt]
        for key in logs:
            self.logger[key] += [logs[key].mean().detach().cpu()]
        return model_loss

    def exploration(self, action, training):
        import ipdb; ipdb.set_trace()
        if training:
            amount = self.cfg.expl_amount
            if self.cfg.expl_decay:
                amount *= 0.5 ** (self.step.float() / self.cfg.expl_decay)
            if self.cfg.expl_min:
                amount = torch.max(self.cfg.expl_min, amount)
            self.logger['expl_amount'] += [amount]
        elif self.cfg.eval_noise:
            amount = self.cfg.eval_noise
        else:
            return action
        import ipdb; ipdb.set_trace()
        if self.cfg.expl == 'additive_gaussian':
          return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        if self.cfg.expl == 'completely_random':
          return tf.random.uniform(action.shape, -1, 1)
        if self.cfg.expl == 'epsilon_greedy':
          indices = tfd.Categorical(0 * action).sample()
          return tf.where(
              tf.random.uniform(action.shape[:1], 0, 1) < amount,
              tf.one_hot(indices, action.shape[-1], dtype=self._float),
              action)
        raise NotImplementedError(self.cfg.expl)

    def policy(self, obs, state, training):
        if state is None:
            latent = self.dynamics.initial(len(obs['image']))
            action = torch.zeros((len(obs['image']), self.act_n)).to(self.cfg.device)
        else:
            latent, action = state
        embed = self.encoder(obs)
        latent, _ = self._dynamics.obs_step(latent, action, embed)
        feat = self.dynamics.get_feat(latent)
        if training:
            action = self.actor(feat).rsample()
        else:
            action = self.actor(feat).mean
        action = torch.tanh(action)
        action = self.exploration(action, training)
        state = (latent, action)
        return action, state

    def get_batch(self):
        bt = time.time()
        batch = next(self.data_iter)
        if self.cfg.use_image:
            batch = nest.map_structure(lambda x: torch.tensor(x).to(self.cfg.device), batch)
            batch['image'] = (batch['image'].permute([0, 1, -1, 2, 3]) / 255.0) - 0.5
            #batch = nest.map_structure(lambda x: x.flatten(0,1), batch)
        else:
            if 'image' in batch: batch.pop('image')
            batch = nest.map_structure(lambda x: torch.tensor(x).to(self.cfg.device), batch)
        self.logger['dt/batch'] += [time.time() - bt]
        return batch

    def run(self):
        if self.cfg.mode == 'dyn':
            self.refresh_dataset()
            start_time = time.time()
            epoch_time = time.time()
            for self.t in itertools.count(1):
                ut = time.time()
                #self.update(batch, log_extra=1)
                batch = self.get_batch()
                self.update(batch, log_extra=self.t%self.cfg.log_n==0)
                self.logger['dt/update'] += [time.time() - ut]
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
        elif self.cfg.mode == 'dream':
            # fill up with initial random data
            self.collect_episode(self.cfg.ep_len, 50)
            #self.refresh_dataset()
            #batch = self.get_batch()
            #self.update(batch, log_extra=0)
            #import ipdb; ipdb.set_trace()
            #num_files = self.cfg.replay_size // (self.cfg.ep_len * self.cfg.num_eps)
            #print(num_files)
            #for _ in range(num_files):