import itertools
import torch
from algo.trainer import Trainer
from copy import deepcopy
from nets import models
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
from nets import models, sacnets
from algo.stats import RunningMeanStd

class SAC(Trainer):
    def __init__(self, cfg, make_env):
        super().__init__(cfg, make_env)
        #self.pi = models.ActionDecoder(self.act_n, cfg, obs_n=self.state_shape[0]).to(cfg.device)
        self.pi = sacnets.SquashedGaussianActor(self.tenv, 'mlp', self.state_shape[0], self.act_n, [256, 256], nn.ReLU, 1.0, cfg).to(cfg.device)
        self.q1 = models.MLP([self.act_n+self.state_shape[0], 256, 256, 1]).to(self.cfg.device)
        self.q2 = models.MLP([self.act_n+self.state_shape[0], 256, 256, 1]).to(self.cfg.device)
        self.target_entropy = -self.act_n
        if cfg.learned_alpha:
            self.log_alpha = torch.nn.Parameter(torch.zeros(1))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.tq1 = deepcopy(self.q1)
        self.tq2 = deepcopy(self.q2)
        for p in self.tq1.parameters(): p.requires_grad = False
        for p in self.tq2.parameters(): p.requires_grad = False
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=self.cfg.pi_lr)
        self.q_optimizer = optim.Adam(self.q_params, lr=self.cfg.vf_lr)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, batch):
        alpha = self.cfg.alpha if not self.cfg.learned_alpha else torch.exp(self.log_alpha).detach().to(self.cfg.device)
        o = batch['state'][:,:-1].flatten(0,1)
        o2 = batch['state'][:,1:].flatten(0,1)
        a = batch['act'][:,:-1].flatten(0,1)
        r = batch['rew'][:,:-1].flatten(0,1)
        c = lambda x,y: torch.cat([x,y],-1)
        d = 0.0

        q1 = self.q1(c(o,a))
        q2 = self.q2(c(o,a))

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.pi(o2)
            #out_pi = self.pi(o2)
            #dist = self.pi.get_dist(out_pi)
            #a2 = dist.sample()
            #logp_a2 = dist.log_prob(a2).detach()
            # Target Q-values
            q1_pi_targ = self.tq1(c(o2, a2))
            q2_pi_targ = self.tq2(c(o2, a2))
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.cfg.gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, batch):
        alpha = self.cfg.alpha if not self.cfg.learned_alpha else torch.exp(self.log_alpha).detach().to(self.cfg.device)
        #o = batch['state'][:,:-1].flatten(0,1)
        o = batch['state'].flatten(0,1)
        c = lambda x,y: torch.cat([x,y],-1)

        pi, logp_pi = self.pi(o)
        #out_pi = self.pi(o)
        #dist = self.pi.get_dist(out_pi)
        #_pi = dist.sample()
        #pi = _pi.clone()
        #pi[_pi > 0.0] -= 0.0001
        #pi[_pi < 0.0] += 0.0001
        #logp_pi = dist.log_prob(pi)

        q1_pi = self.q1(c(o, pi))
        q2_pi = self.q2(c(o, pi))
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        if self.cfg.learned_alpha:
            loss_alpha = (-1.0 * (torch.exp(self.log_alpha).to(self.cfg.device) * (logp_pi + self.target_entropy).detach())).mean()
        else:
            loss_alpha = 0.0

        return loss_pi, loss_alpha, pi_info

    def policy(self, obs, state=None, training=True):
        action, _ = self.pi(torch.tensor(obs['state']).to(self.cfg.device))
        #action = self.pi.get_dist(self.pi(torch.tensor(obs['state']).to(self.cfg.device))).sample()
        #action = torch.clamp(action, -0.9995, 0.9995)
        return action, state

    def update(self, batch, log_extra=False):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger['LossQ'] += [loss_q.detach().cpu()]
        for key in q_info:
            self.logger[key] += [q_info[key]]

        # Freeze Q-networks so you don'self waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, loss_alpha, pi_info = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optimizer.step()
        # and optionally the alpha
        if self.cfg.learned_alpha:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
            self.logger['LossAlpha'] += [loss_alpha.detach().cpu()]
            self.logger['Alpha'] += [torch.exp(self.log_alpha.detach().cpu())]

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger['LossPi'] += [loss_pi.detach().cpu()]
        for key in pi_info:
            self.logger[key] += [pi_info[key]]

        # Finally, update target networks by polyak averaging.
        # TODO: try the update every 100 steps method
        #with torch.no_grad():
        #    for p, p_targ in zip(self.q1.parameters(), self.tq1.parameters()):
        #        p_targ.data.mul_(self.cfg.polyak); p_targ.data.add_((1 - self.cfg.polyak) * p.data)
        #    for p, p_targ in zip(self.q2.parameters(), self.tq2.parameters()):
        #        p_targ.data.mul_(self.cfg.polyak); p_targ.data.add_((1 - self.cfg.polyak) * p.data)

    def run(self):
        self.start_time = time.time()
        self.dt_time = time.time()
        # fill up with initial random data
        #for i in range(self.cfg.warmup):
        self.collect_episode(self.cfg.ep_len, 10, mode='random')

        for self.t in itertools.count():
            self.refresh_dataset()
            for j in range(2000):
                batch = self.get_batch()
                self.update(batch, log_extra=j==0 and self.t%5==0)
                ## Finally, update target network
                if j % 100 == 0:
                    with torch.no_grad():
                        for p, p_targ in zip(self.q1.parameters(), self.tq1.parameters()): p_targ.data[:] = p.data[:]
                        for p, p_targ in zip(self.q2.parameters(), self.tq2.parameters()): p_targ.data[:] = p.data[:]
            self.collect_episode(self.cfg.ep_len, 10, mode='policy')
            if self.t % 1 == 0:
                self.logger_dump()
        #num_files = self.cfg.replay_size // (self.cfg.ep_len * self.cfg.num_eps)
        #print(num_files)
        #for _ in range(num_files):