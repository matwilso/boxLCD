import itertools
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import scipy.signal
import torch
import yaml

# from research.nets.flat_everything import FlatEverything
from jax.tree_util import tree_map, tree_multimap
from torch.optim import Adam

import boxLCD
from boxLCD import env_map
from research import utils, wrappers
from research.define_config import args_type, config, env_fn
from research.nets import net_map
from research.nets.autoencoders.bvae import BVAE
from research.rl.buffers import OGRB, ReplayBuffer
from research.rl.sacnets import ActorCritic

from ._base import RLAlgo


class SAC(RLAlgo):
    def __init__(self, G):
        super().__init__(G)
        # Create actor-critic module and target networks
        self.ac = ActorCritic(
            self.obs_space, self.act_space, self.goal_key, G=self.G
        ).to(self.G.device)
        self.ac_targ = deepcopy(self.ac)
        var_counts = tuple(
            utils.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2]
        )
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)
        self.sum_count = sum(var_counts)
        print(f'total: {self.sum_count}')

        # Freeze target networks with respect to optimizers (only updte via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.ac.q1.parameters(), self.ac.q2.parameters()
        )

        # Experience buffer
        self.buf = ReplayBuffer(
            self.G, obs_space=self.obs_space, act_space=self.act_space
        )

        # Set up optimizers for policy and q-function
        self.q_optimizer = Adam(
            self.q_params, lr=self.G.lr, betas=(0.9, 0.999), eps=1e-8
        )
        self.pi_optimizer = Adam(
            self.ac.pi.parameters(), lr=self.G.lr, betas=(0.9, 0.999), eps=1e-8
        )
        if self.G.learned_alpha:
            self.alpha_optimizer = Adam([self.ac.log_alpha], lr=self.G.alpha_lr)

        self.test_agent(-1)
        if self.G.lenv:
            self.test_agent(-1, use_lenv=True)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        q_info = {}
        alpha = (
            self.G.alpha
            if not self.G.learned_alpha
            else torch.exp(self.ac.log_alpha).detach()
        )
        o, a, r, o2, d = (
            data['obs'],
            data['act'],
            data['rew'],
            data['obs2'],
            data['done'],
        )
        if not self.G.use_done:
            d = 0
        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, ainfo = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.G.gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info['q1/mean'] = q1.mean()
        q_info['q2/mean'] = q2.mean()
        q_info['q1/min'] = q1.min()
        q_info['q1/max'] = q1.max()
        q_info['batchR/mean'] = r.mean()
        q_info['batchR/min'] = r.min()
        q_info['batchR/max'] = r.max()
        q_info['residual_variance'] = (q1 - backup).var() / backup.var()
        q_info['target_min'] = backup.min()
        q_info['target_max'] = backup.max()
        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        alpha = (
            self.G.alpha
            if not self.G.learned_alpha
            else torch.exp(self.ac.log_alpha).detach()
        )
        o = data['obs']
        pi, logp_pi, ainfo = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(
            LogPi=logp_pi.mean().detach().cpu(),
            action_abs=ainfo['mean'].abs().mean().detach().cpu(),
            action_std=ainfo['std'].mean().detach().cpu(),
        )

        if self.G.learned_alpha:
            loss_alpha = (
                -1.0
                * (
                    torch.exp(self.ac.log_alpha)
                    * (logp_pi + self.ac.target_entropy).detach()
                )
            ).mean()
            # loss_alpha = -(self.ac.log_alpha * (logp_pi + self.ac.target_entropy).detach()).mean()
        else:
            loss_alpha = 0.0

        return loss_pi, loss_alpha, pi_info

    def update(self, data):
        # TODO: optimize this by not requiring the items right away.
        # i think this might be blockin for pytorch to finish some computations

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.logger['Q/grad_norm'] += [
            utils.compute_grad_norm(self.ac.q1.parameters()).detach().cpu()
        ]
        self.q_optimizer.step()

        # Record things
        self.logger['LossQ'] += [loss_q.detach().cpu()]
        for key in q_info:
            self.logger[key] += [q_info[key].detach().cpu()]

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False
            if 'vae' in self.G.net:
                for p in self.ac.preproc.parameters():
                    p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, loss_alpha, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.logger['Pi/grad_norm'] += [
            utils.compute_grad_norm(self.ac.pi.parameters()).detach().cpu()
        ]
        self.pi_optimizer.step()
        # and optionally the alpha
        if self.G.learned_alpha:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
            self.logger['LossAlpha'] += [loss_alpha.detach().cpu()]
            self.logger['Alpha'] += [torch.exp(self.ac.log_alpha.detach().cpu())]

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger['LossPi'] += [loss_pi.detach().cpu()]
        for key in pi_info:
            self.logger[key] += [pi_info[key]]

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.G.polyak)
                p_targ.data.add_((1 - self.G.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        o = {
            key: torch.as_tensor(1.0 * val, dtype=torch.float32).to(self.G.device)
            for key, val in o.items()
        }
        act = self.ac.act(o, deterministic)
        if not self.G.lenv:
            act = act.cpu().numpy()
        return act

    def get_av(self, o, deterministic=False):
        with torch.no_grad():
            o = {
                key: torch.as_tensor(1.0 * val, dtype=torch.float32).to(self.G.device)
                for key, val in o.items()
            }
            a, _, ainfo = self.ac.pi(o, deterministic, False)
            q = self.ac.value(o, a)
        if not self.G.lenv:
            a = a.cpu().numpy()
        return a, q

    def run_firehose(self):
        epoch_time = start_time = time.time()
        # TODO: try this with backpropping through stuff
        o = self.env.reset(np.arange(self.G.num_envs))
        for itr in itertools.count(1):
            a = self.get_action(o).detach()
            o2, rew, done, info = self.env.step(a)
            batch = {'obs': o, 'act': a, 'rew': rew, 'obs2': o2, 'done': done}
            batch = tree_map(lambda x: x.detach(), batch)
            self.update(batch)
            o = o2
            # success = info['success']
            # self.env._reset_goals(success)
            # print(itr)
            if itr % 200 == 0:
                o = self.env.reset()
            if itr % self.G.log_n == 0:
                epoch = itr // self.G.log_n
                if epoch % self.G.test_n == 0:
                    self.test_agent(itr)
                    if self.G.lenv:
                        self.test_agent(itr, use_lenv=True)
                # Log info about epoch
                print('=' * 30)
                print('Epoch', epoch)
                self.logger['var_count'] = [self.sum_count]
                self.logger['dt'] = dt = time.time() - epoch_time
                for key in self.logger:
                    val = np.mean(self.logger[key])
                    self.writer.add_scalar(key, val, itr)
                    print(key, val)
                self.writer.flush()
                print('Time', time.time() - start_time)
                print('dt', dt)
                print(self.G.logdir)
                print(self.G.full_cmd)
                print('=' * 30)
                self.logger = defaultdict(lambda: [])
                epoch_time = time.time()
                with open(self.G.logdir / 'hps.yaml', 'w') as f:
                    yaml.dump(self.G, f, width=1000)

    def run(self):
        epoch = -1
        # Prepare for interaction with environment
        epoch_time = self.start_time = time.time()
        if self.G.lenv:
            o, ep_ret, ep_len = (
                self.env.reset(),
                torch.zeros(self.G.num_envs).to(self.G.device),
                torch.zeros(self.G.num_envs).to(self.G.device),
            )
            success = torch.zeros(self.G.num_envs).to(self.G.device)
            time_to_succ = self.G.ep_len * torch.ones(self.G.num_envs).to(self.G.device)
            pf = th
        else:
            o, ep_ret, ep_len = (
                self.env.reset(),
                np.zeros(self.G.num_envs),
                np.zeros(self.G.num_envs),
            )
            success = np.zeros(self.G.num_envs, dtype=np.bool)
            time_to_succ = self.G.ep_len * np.ones(self.G.num_envs)
            pf = np
        # Main loop: collect experience in venv and update/log each epoch
        for itr in range(1, self.G.total_steps + 1):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if itr > self.G.start_steps:
                with utils.Timer(self.logger, 'action'):
                    o = {key: val for key, val in o.items()}
                    a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            # Step the venv
            o2, r, d, info = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d[ep_len == self.G.ep_len] = False
            success = pf.logical_or(success, d)
            time_to_succ = pf.minimum(
                time_to_succ, self.G.ep_len * ~success + ep_len * success
            )
            # print(time_to_succ)

            # Store experience to replay buffer
            trans = {'act': a, 'rew': r, 'done': d}
            for key in o:
                trans[f'o:{key}'] = o[key]
            for key in o2:
                trans[f'o2:{key}'] = o2[key]
            self.buf.store_n(trans)

            # Super critical, easy to overlook step: make sure to updte
            # most recent observation!
            o = o2

            # End of trajectory handling
            if self.G.lenv:
                done = torch.logical_or(d, ep_len == self.G.ep_len)
                dixs = torch.nonzero(done)

                def proc(x):
                    return x.cpu().float()

            else:
                done = np.logical_or(d, ep_len == self.G.ep_len)
                dixs = np.nonzero(done)[0]

                def proc(x):
                    return x

            if len(dixs) == self.G.num_envs or (not self.G.lenv and self.G.succ_reset):
                # if not self.G.lenv or len(dixs) == self.G.num_envs:
                for idx in dixs:
                    self.logger['EpRet'] += [proc(ep_ret[idx])]
                    self.logger['EpLen'] += [proc(ep_len[idx])]
                    self.logger['success_rate'] += [proc(success[idx])]
                    self.logger['time_to_succ'] += [proc(time_to_succ[idx])]
                    ep_ret[idx] = 0
                    ep_len[idx] = 0
                    success[idx] = 0
                    time_to_succ[idx] = self.G.ep_len
                    # self.logger['success_rate'] += [proc(d[idx])]
                    # self.logger['success_rate_nenv'] += [proc(d[idx])**(1/self.G.num_envs)]
                if len(dixs) != 0:
                    if not self.G.autoreset:
                        o = self.env.reset(dixs)
                    if self.G.lenv:
                        assert (
                            len(dixs) == self.G.num_envs
                        ), "the learned env needs the envs to be in sync in terms of history.  you can't easily reset one without resetting the others. it's hard"
                    else:
                        assert (
                            self.env.shared_memory
                        ), "i am not sure if this works when you don't do shared memory. it would need to be tested. something like the comment below"
                    # o = tree_multimap(lambda x,y: ~done[:,None]*x + done[:,None]*y, o, reset_o)

            # Updte handling
            if itr >= self.G.update_after and itr % self.G.update_every == 0:
                for j in range(int(self.G.update_every * 1.0)):
                    with utils.Timer(self.logger, 'sample_batch'):
                        batch = self.buf.sample_batch(self.G.bs)
                    with utils.Timer(self.logger, 'updte'):
                        self.update(data=batch)

            # End of epoch handling
            if itr % self.G.log_n == 0:
                epoch = itr // self.G.log_n

                # Save model
                if (epoch % self.G.save_freq == 0) or (itr == self.G.total_steps):
                    torch.save(self.ac, self.G.logdir / 'weights.pt')

                if (self.G.logdir / 'pause.marker').exists():
                    import ipdb

                    ipdb.set_trace()

                # Test the performance of the deterministic version of the agent.
                if epoch % self.G.test_n == 0:
                    with utils.Timer(self.logger, 'test_agent'):
                        self.test_agent(itr)
                        if self.G.lenv:
                            self.test_agent(itr, use_lenv=True)

                    if 'vae' in self.G.net:
                        test_batch = self.buf.sample_batch(8)
                        self.ac.preproc.evaluate(self.writer, test_batch['obs'], itr)

                # Log info about epoch
                self.logger['var_count'] = [self.sum_count]
                self.logger['dt'] = dt = time.time() - epoch_time
                self.logger['env_interactions'] = env_interactions = (
                    itr * self.G.num_envs
                )
                self.logger = utils.dump_logger(self.logger, self.writer, itr, self.G)
                epoch_time = time.time()
