import time

import numpy as np
import scipy.signal
import torch

# from research.nets.flat_everything import FlatEverything
from jax.tree_util import tree_map, tree_multimap
from torch.optim import Adam

import boxLCD
from boxLCD import env_map
from research import utils, wrappers
from research.define_config import env_fn
from research.rl.buffers import OGRB, PPOBuffer, ReplayBuffer
from research.rl.pponets import ActorCritic

from ._base import TN, RLAlgo


class PPO(RLAlgo):
    def __init__(self, G):
        super().__init__(G)
        # Create actor-critic module and target networks
        self.ac = ActorCritic(self.obs_space, self.act_space, self.goal_key, G=G).to(
            G.device
        )
        # Experience buffer
        self.buf = PPOBuffer(
            G,
            obs_space=self.obs_space,
            act_space=self.act_space,
            size=G.num_envs * G.steps_per_epoch,
        )

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(
            utils.count_vars(module) for module in [self.ac.pi, self.ac.v]
        )
        print('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)
        self.sum_count = sum(var_counts)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(
            self.ac.pi.parameters(), lr=G.pi_lr, betas=(0.9, 0.999), eps=1e-8
        )
        self.vf_optimizer = Adam(
            self.ac.v.parameters(), lr=G.vf_lr, betas=(0.9, 0.999), eps=1e-8
        )

        self.test_agent(-1)
        if self.G.lenv:
            self.test_agent(-1, use_lenv=True)

    def get_av(self, o):
        return self.ac.step(o)[:2]

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = (
            torch.clamp(ratio, 1 - self.G.clip_ratio, 1 + self.G.clip_ratio) * adv
        )
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        # Useful extra info
        approx_kl = (logp_old - logp).mean().cpu()
        ent = pi.entropy().mean().cpu()
        clipped = ratio.gt(1 + self.G.clip_ratio) | ratio.lt(1 - self.G.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().cpu()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self, data):
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.cpu()
        v_l_old = self.compute_loss_v(data).cpu()

        # Train policy with multiple steps of gradient descent
        for i in range(self.G.train_pi_iters):
            idxs = np.random.randint(0, data['act'].shape[0], self.G.bs)
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(tree_map(lambda x: x[idxs], data))
            kl = pi_info['kl']
            # if kl > 1.5 * self.G.target_kl:
            #  break
            loss_pi.backward()
            self.pi_optimizer.step()

        self.logger['StopIter'] += [i]

        # Value function learning
        for i in range(self.G.train_v_iters):
            idxs = np.random.randint(0, data['act'].shape[0], self.G.bs)
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(tree_map(lambda x: x[idxs], data))
            loss_v.backward()
            self.vf_optimizer.step()

        # Log changes from updte
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger['LossPi'] += [pi_l_old.detach().cpu()]
        self.logger['LossV'] += [v_l_old.detach().cpu()]
        self.logger['KL'] += [kl.detach().cpu()]
        self.logger['Entropy'] += [ent.detach().cpu()]
        self.logger['ClipFrac'] += [cf.detach().cpu()]
        self.logger['DeltaLossPi'] += [loss_pi.detach().cpu() - pi_l_old.detach().cpu()]
        self.logger['DeltaLossV'] += [loss_v.detach().cpu() - v_l_old.detach().cpu()]

    def run_firehose(self):
        """run w/o leaving GPU"""

    def run(self):
        # Prepare for interaction with environment
        epoch = -1
        epoch_time = self.start_time = time.time()

        if self.G.lenv:
            o, ep_ret, ep_len = (
                self.env.reset(np.arange(self.G.num_envs)),
                torch.zeros(self.G.num_envs).to(self.G.device),
                np.zeros(self.G.num_envs),
            )  # .to(G.device)
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
        # Main loop: collect experience in venv and updte/log each epoch
        for itr in range(1, self.G.total_steps + 1):
            with utils.Timer(self.logger, 'action'):
                o = {key: val for key, val in o.items()}
                a, v, logp = self.ac.step(o)
            # Step the venv
            with utils.Timer(self.logger, 'step'):
                next_o, r, d, info = self.env.step(a)  # , self.logger)
            ep_ret += r
            ep_len += 1

            # store
            trans = {'act': a, 'rew': r, 'val': v, 'logp': logp}
            for key in o:
                trans[f'o:{key}'] = o[key]
            if self.G.lenv:

                def fx(x):
                    if isinstance(x, np.ndarray):
                        return x
                    else:
                        return x.detach().cpu().numpy()

                trans = tree_map(fx, trans)
            self.buf.store_n(trans)

            o = next_o

            if self.G.lenv:
                d = d.cpu().numpy()

                def proc(x):
                    return x.cpu().float()

            else:

                def proc(x):
                    return x

            timeout = ep_len == self.G.ep_len
            terminal = np.logical_or(d, timeout)
            epoch_ended = itr % self.G.steps_per_epoch == 0
            terminal_epoch = np.logical_or(terminal, epoch_ended)
            timeout_epoch = np.logical_or(timeout, epoch_ended)
            mask = ~timeout_epoch
            if self.G.learned_rew:
                # self.logger['preproc_rew'] += [info['preproc_rew'].mean()]
                self.logger['learned_rew'] += [info['learned_rew'].mean()]
                self.logger['og_rew'] += [info['og_rew'].mean()]
                self.logger['goal_delta'] += [info['goal_delta'].mean()]
                self.logger['rew_delta'] += [info['rew_delta'].mean()]
            # if trajectory didn't reach terminal state, bootstrap value target
            _, v, _ = self.ac.step(o)
            v[mask] *= 0
            self.buf.finish_paths(np.nonzero(terminal_epoch)[0], v)
            for idx in np.nonzero(terminal_epoch)[0]:
                self.logger['EpRet'] += [proc(ep_ret[idx])]
                self.logger['EpLen'] += [ep_len[idx]]
                ep_ret[idx] = 0
                ep_len[idx] = 0

            if epoch_ended:
                if (self.G.logdir / 'pause.marker').exists():
                    import ipdb

                    ipdb.set_trace()
                epoch = itr // self.G.steps_per_epoch
                self.update(self.buf.get())
                with utils.Timer(self.logger, 'test_agent'):
                    self.test_agent(itr)
                    if self.G.lenv:
                        self.test_agent(itr, use_lenv=True)

                # save it
                self.ac.save(self.G.logdir)
                self.logger['var_count'] = [self.sum_count]
                self.logger['dt'] = dt = time.time() - epoch_time
                self.logger['env_interactions'] = env_interactions = (
                    itr * self.G.num_envs
                )
                self.logger = utils.dump_logger(self.logger, self.writer, itr, self.G)
                epoch_time = time.time()
