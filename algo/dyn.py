import time
import itertools
import torch
from copy import deepcopy
from jax.tree_util import tree_multimap
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tensorflow import nest
import numpy as np
from algo.trainer import Trainer
from torch import distributions
import utils
#from torch.cuda import amp
from nets import models
from algo.stats import RunningMeanStd

def fvmap(f):
    """fakes vmap by just reshaping"""
    def _thunk(x):
        # TODO: add support for nest.mapping this over a dict of stuff.
        preshape = x.shape
        x = f(x.flatten(0, 1))
        return x.reshape(preshape[:2]+x.shape[1:])
    return _thunk

def pass_clamp(b, bound):
    nb = b.clone()
    mask = (b<-bound).__or__(b>bound)
    nb[mask] = b[mask] - (torch.sign(b[mask])*(torch.abs(b[mask]) - bound)).detach()
    return nb

class Dyn(Trainer, nn.Module):
    def __init__(self, cfg, make_env):
        nn.Module.__init__(self)
        Trainer.__init__(self, cfg, make_env)
        if cfg.use_image:
            self.encoder = models.ConvEncoder(cfg).to(cfg.device)
            self.decoder = models.ConvDecoder(cfg).to(cfg.device)
            self.skey = 'image'
        else:
            self.encoder = models.DenseEncoder(self.state_shape[0], cfg).to(cfg.device)
            self.decoder = models.DenseDecoder(self.state_shape[0], cfg).to(cfg.device)
            self.skey = 'state'
        self.dynamics = models.RSSM(self.act_n, cfg).to(cfg.device)
        self.latent_fwds = models.LatentFwds(self.act_n, cfg).to(cfg.device)
        if self.cfg.reward_mode == 'normal':
            self.reward = models.DenseDecoder(1, cfg).to(cfg.device)

        # dreamer stuff
        self.value = models.DenseDecoder(1, cfg).to(cfg.device)
        self.targ_value = deepcopy(self.value)
        for p in self.targ_value.parameters():
            p.requires_grad = False
        self.actor = models.ActionDecoder(self.act_n, cfg).to(cfg.device)
        self.model_params = itertools.chain(self.encoder.parameters(), self.decoder.parameters(), self.dynamics.parameters(), self.reward.parameters())
        self.model_optimizer = optim.Adam(self.model_params, lr=self.cfg.dyn_lr)
        self.lds_optimizer = optim.Adam(self.latent_fwds.parameters(), lr=self.cfg.lds_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.pi_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.cfg.vf_lr)
        self.actdist = lambda x: self.actor.get_dist(self.actor(x))

        # TODO: try out just autoencoder first
        self.venc = fvmap(self.encoder)
        #params = list(self.encoder.parameters())
        #params = nest.map_structure(lambda x: x[None].repeat(3, *([1]*len(x.shape))), params)
        #self.scaler = amp.GradScaler()
        #self.encoder._parameters = params
        if self.cfg.obs_stats:
            self.obs_rms = RunningMeanStd(shape=self.state_shape[0])

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
        def get_state(x):
            x = np.array(x.detach().cpu())
            return x if not self.cfg.obs_stats else (x*(1.0*self.obs_rms.var**0.5) + self.obs_rms.mean)
        truth = np.zeros([50, 3, 64, self.cfg.env_wh_ratio*64])
        for i in range(50):
            truth[i] = self.tenv.env.visualize_obs(get_state(data['state'][0][i])).transpose(2,0,1) / 255.0

        recon = obs_pred.mean
        recon = recon.reshape((self.cfg.bs, 50)+recon.shape[1:])[0]
        init, _ = self.dynamics.observe(embed[:5, :5], data['act'][:5, :5])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self.dynamics.imagine(data['act'][:5, 5:], init)
        std = torch.tensor(self.obs_rms.var**0.5).to(self.cfg.device) if self.cfg.obs_stats else None
        openl = self.decoder(self.dynamics.get_feat(prior).flatten(0,1), std).mean
        openl = openl.reshape((5, 45) + openl.shape[1:])

        model = np.zeros([50, 3, 64, self.cfg.env_wh_ratio*64])
        for i in range(5):
            model[i] = self.tenv.env.visualize_obs(get_state(recon[i])).transpose(2,0,1) / 255.0
        for i in range(5, 50):
            model[i] = self.tenv.env.visualize_obs(get_state(openl[0,i-5])).transpose(2,0,1) / 255.0

        error = (model - truth + 1) / 2
        openl = np.concatenate([truth, model, error], 3)[None]
        self.writer.add_video('agent/openl', openl, self.t, fps=20)

    def imagine_ahead(self, post):
        start = {k: v.flatten(0,1) for k, v in post.items()}
        actions = {key: torch.zeros([50*50, self.act_n]).to(self.cfg.device) for key in ['action', 'mean', 'std']}
        # TODO: make policy autoregressive so we get better propagation.
        def func(prev, _):
            prev = prev[0]
            act_stats = self.actor(self.dynamics.get_feat(prev).detach())
            poo = self.actor.get_dist(act_stats).rsample()
            new_poo = poo.clone()
            new_poo[poo > 0.0] -= 0.0001
            new_poo[poo < 0.0] += 0.0001
            #poo = pass_clamp(poo, 0.9995)
            #poo = self.actor(self.dynamics.get_feat(prev)).rsample()
            return self.dynamics.img_step(prev, new_poo), {'action': new_poo, **act_stats}
        outs = utils.static_scan(func, torch.arange(0, self.cfg.horizon).to(self.cfg.device), (start, actions))
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
        obs_pred = self.decoder(feat.flatten(0,1), std=batch['std'] if self.cfg.obs_stats else None)
        prior_dist = self.dynamics.get_dist(prior)
        post_dist = self.dynamics.get_dist(post)

        recon_loss = -obs_pred.log_prob(obs.flatten(0,1)).mean()
        transition_loss = -0.08*prior_dist.log_prob(post['stoch']).mean()
        latent_ent_loss = +0.02*post_dist.log_prob(post['stoch']).mean()

        model_loss = recon_loss +  transition_loss + latent_ent_loss
        #self.scaler.scale(model_loss).backward()
        feat = self.dynamics.get_feat(post).detach()
        if self.cfg.reward_mode == 'normal':
            #rew_loss = -self.reward(feat).log_prob(batch['rew'])
            rew_loss = -self.reward(feat).log_prob(self.cfg.rew_weight*batch['rew'])
            logs['model/rew_loss'] = rew_loss
            logs['reward/batch'] = batch['rew'].mean()
            model_loss += rew_loss.mean()
            #model_loss += self.cfg.rew_weight*rew_loss.mean()
        elif self.cfg.reward_mode == 'explore':
            lds = self.latent_fwds(torch.cat([feat[...,:-30], batch['act']], -1))
            lds_loss = (0.5*(feat[:,1:].detach() - lds[:,:,:-1])**2).mean()
            logs['model/lds_loss'] = lds_loss
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
        if self.cfg.mode == 'dream':
            # ACTOR UPDATE
            self.actor_optimizer.zero_grad()
            post = {k: v.detach() for k,v in post.items()}
            imag_feat, actions = self.imagine_ahead(post)
            lds = self.latent_fwds(torch.cat([imag_feat[...,:-30], actions['action']], -1))
            if self.cfg.reward_mode == 'normal':
                reward = self.reward(imag_feat.detach()).mean[...,0]
            else:
                reward = self.cfg.rew_weight*lds.var(0).mean(-1).detach()
            pcont = self.cfg.gamma * torch.ones_like(reward)
            value = self.value(imag_feat).mean[...,0]
            returns = utils.lambda_return(reward[:-1], value[:-1], pcont[:-1], bootstrap=value[-1], lambda_=self.cfg.lam, axis=0)
            discount = torch.cumprod(torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0).detach()
            vt_gam = discount * returns
            act_dist = self.actor.get_dist(actions)
            act_logp = act_dist.log_prob(actions['action'])
            reinforce_loss = -(act_logp[:-1]*((vt_gam - value[1:]).detach())).mean()
            dyn_back_loss = -vt_gam.mean()

            act_ent_loss = (self.actdist(feat.detach()).log_prob(batch['act'])).mean()

            # TODO: try both
            #actor_loss = dyn_back_loss + self.cfg.act_ent_weight*act_ent_loss
            actor_loss = 0.9*reinforce_loss + 0.1*dyn_back_loss  +self.cfg.act_ent_weight*act_ent_loss
            #actor_loss = -(vt_gam).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            # VALUE UPDATE
            self.value_optimizer.zero_grad()
            targ_value = self.targ_value(imag_feat.detach()).mean[...,0]
            targ_returns = utils.lambda_return(reward[:-1], targ_value[:-1], pcont[:-1], bootstrap=targ_value[-1], lambda_=self.cfg.lam, axis=0)
            target = targ_returns.detach()
            value_pred = self.value(imag_feat.detach())
            logp = value_pred.log_prob(torch.cat([target, torch.ones((1,)+target.shape[1:]).to(self.cfg.device)])[...,None])[:-1]
            value_loss = -(discount * logp).mean()
            self.value_optimizer.step()

            logs['value_loss'] = value_loss
            logs['actor_loss'] = actor_loss
            logs['actor/reinforce_loss'] = reinforce_loss
            logs['actor/dyn_back_loss'] = dyn_back_loss
            logs['actor/act_ent_loss'] = act_ent_loss
            logs['reward/pred'] = reward.mean()

        logs['model/recon_loss'] = recon_loss
        logs['model/transition_loss'] = transition_loss
        logs['model/latent_ent_loss'] = latent_ent_loss
        logs['model_loss'] = model_loss
        if log_extra:
            logs['prior_ent'] = prior_dist.entropy()
            logs['post_ent'] = post_dist.entropy()
            #self.logger['model_grad_norm'] = post_dist.sentropy()
            lt = time.time()
            if self.cfg.use_image:
                self.image_summaries(batch, embed, obs_pred)
            else:
                self.state_summaries(batch, embed, obs_pred)
            self.logger['dt/summary'] += [time.time()-lt]
        for key in logs:
            self.logger[key] += [logs[key].mean().detach().cpu()]

        ## Finally, update target networks by polyak averaging.
        #with torch.no_grad():
        #    for p, p_targ in zip(self.value.parameters(), self.targ_value.parameters()):
        #        p_targ.data.mul_(self.cfg.polyak)
        #        p_targ.data.add_((1 - self.cfg.polyak) * p.data)
        return model_loss

    def policy(self, obs, state, training):
        if state is None:
            latent = self.dynamics.initial(len(obs[self.skey]))
            action = torch.zeros((len(obs[self.skey]), self.act_n)).to(self.cfg.device)
        else:
            latent, action = state
        embed = self.encoder(torch.tensor(obs[self.skey]).to(self.cfg.device))
        latent, _ = self.dynamics.obs_step(latent, action, embed)
        feat = self.dynamics.get_feat(latent)
        if training:
            action = self.actor.get_dist(self.actor(feat)).sample()
            action = torch.clamp(action, -0.9995, 0.9995)
        else:
            action = self.actor(feat)['mean']
        state = (latent, action)
        return action, state

    def run(self):
        self.start_time = time.time()
        self.dt_time = time.time()
        if self.cfg.mode == 'dyn':
            self.refresh_dataset()
            for self.t in itertools.count(1):
                ut = time.time()
                #self.update(batch, log_extra=1)
                batch = self.get_batch()
                self.update(batch, log_extra=self.t%self.cfg.log_n==0)
                self.logger['dt/update'] += [time.time() - ut]
                if self.t % self.cfg.log_n == 0:
                    self.logger_dump()
        elif self.cfg.mode == 'dream':
            # fill up with initial random data
            self.collect_episode(self.cfg.ep_len, 1, mode='random')

            for self.t in range(self.cfg.total_steps):
                self.refresh_dataset()
                for j in range(int(100*self.cfg.update_rate)):
                    batch = self.get_batch()
                    self.update(batch, log_extra=j==0 and self.t%5==0)
                    # Finally, update target network
                    with torch.no_grad():
                        for p, p_targ in zip(self.value.parameters(), self.targ_value.parameters()):
                            p_targ.data[:] = p.data[:]

                self.collect_episode(self.cfg.ep_len, 1, mode='policy')
                if self.t % 1 == 0:
                    self.logger_dump()
            #num_files = self.cfg.replay_size // (self.cfg.ep_len * self.cfg.num_eps)
            #print(num_files)
            #for _ in range(num_files):