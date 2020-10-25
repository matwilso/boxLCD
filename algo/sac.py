import torch
from algo.base import Trainer

class SAC(Trainer):
    def __init__(self, cfg, make_env):
        super().__init__(cfg, make_env)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        alpha = self.cfg.alpha if not self.cfg.learned_alpha else torch.exp(self.ac.log_alpha).detach()
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
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
    def compute_loss_pi(self, data):
        alpha = self.cfg.alpha if not self.cfg.learned_alpha else torch.exp(self.ac.log_alpha).detach()
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        if self.cfg.learned_alpha:
            loss_alpha = (-1.0 * (torch.exp(self.ac.log_alpha) * (logp_pi + self.ac.target_entropy).detach())).mean()
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
        loss_pi, loss_alpha, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        # and optionally the alpha
        if self.cfg.learned_alpha:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
            self.logger['LossAlpha'] += [loss_alpha.detach().cpu()]
            self.logger['Alpha'] += [torch.exp(self.ac.log_alpha.detach().cpu())]

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger['LossPi'] += [loss_pi.item()]
        for key in pi_info:
            self.logger[key] += [pi_info[key]]

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.cfg.polyak)
                p_targ.data.add_((1 - self.cfg.polyak) * p.data)

    def run(self):
        # Prepare for interaction with environment
        total_steps = self.cfg.steps_per_epoch * self.cfg.epochs
        epoch_time = start_time = time.time()
        o, ep_ret, ep_len = self.venv.reset(), np.zeros(self.cfg.num_envs), np.zeros(self.cfg.num_envs)
        # Main loop: collect experience in venv and update/log each epoch
        for self.t in range(total_steps):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if self.t > self.cfg.start_steps:
                a = self.get_action(o)
            else:
                a = self.venv.action_space.sample()

            # Step the venv
            o2, r, d, _ = self.venv.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            time_horizon = ep_len==self.cfg.max_ep_len
            d[ep_len==self.cfg.max_ep_len] = False

            # Store experience to replay buffer
            self.replay_buffer.store_n({'obs': o, 'act': a, 'rew': r, 'obs2': o2, 'done': d})

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            done = np.logical_or(d, ep_len == self.cfg.max_ep_len)
            for idx in np.nonzero(done)[0]:
                self.logger['EpRet'] += [ep_ret[idx]]
                self.logger['EpLen'] += [ep_len[idx]]
                ep_ret[idx] = 0
                ep_len[idx] = 0

            # Update handling
            if self.t >= self.cfg.update_after and self.t % self.cfg.update_every == 0:
                for j in range(int(self.cfg.update_every*1.5)):
                    batch = self.replay_buffer.sample_batch(self.cfg.bs)
                    self.update(data=batch)

            # End of epoch handling
            if (self.t+1) % self.cfg.steps_per_epoch == 0:
                epoch = (self.t+1) // self.cfg.steps_per_epoch

                # Save model
                if (epoch % self.cfg.save_freq == 0) or (epoch == self.cfg.epochs):
                    pass

                # Test the performance of the deterministic version of the agent.
                if epoch % 5 == 0:
                    self.test_agent(video=epoch%5==0)

                # Log info about epoch
                print('='*30)
                print('Epoch', epoch)
                self.logger['var_count'] = [self.sum_count]
                for key in self.logger:
                    val = np.mean(self.logger[key])
                    self.writer.add_scalar(key, val, self.t)
                    print(key, val)
                if self.cfg.net == 'router':
                    self.writer.add_image('Pi/i_route', self.ac.pi.net.get_iperm()[None], self.t)
                    self.writer.add_image('Pi/o_route', self.ac.pi.net.get_operm()[None], self.t)
                    self.writer.add_image('Q1/i_route', self.ac.q1.get_iperm()[None], self.t)
                    self.writer.add_image('Q2/i_route', self.ac.q2.get_iperm()[None], self.t)
                self.writer.flush()
                print('TotalEnvInteracts', self.t*self.cfg.num_envs)
                print('Time', time.time()-start_time)
                print('dt', time.time()-epoch_time)
                print(self.logpath)
                print(self.cfg.full_cmd)
                print('='*30)
                self.logger = defaultdict(lambda: [])
                epoch_time = time.time()