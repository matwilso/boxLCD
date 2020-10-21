from datetime import datetime
import PIL
from collections import defaultdict
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from gym.vector.async_vector_env import AsyncVectorEnv
from rb import ReplayBuffer
from cfg import define_cfg, args_type, env_fn, make_env
import utils
from nets.sacnets import ActorCritic
from functools import partial

# TODO: add support for saving weights to file and loading
def sac(cfg):
    print(cfg.full_cmd)
    seed = cfg.seed
    # Set up logger and save configuration
    logger = defaultdict(lambda: [])
    timestamp = datetime.now().strftime('%Y%m%dT-%H-%M-%S')
    logpath = cfg.logdir/f'{cfg.env}/{cfg.exp_name}-{timestamp}'
    writer = SummaryWriter(logpath)

    torch.manual_seed(seed)
    np.random.seed(seed)
    venv = AsyncVectorEnv([make_env(cfg, i) for i in range(cfg.num_envs)]) # vector env
    tenv = make_env(cfg, seed)() # test env
    obs_dim = tenv.observation_space.shape
    act_dim = tenv.action_space.shape[0]
    obs = venv.reset()
    import ipdb; ipdb.set_trace()

    # Create actor-critic module and target networks
    ac = ActorCritic(tenv, tenv.observation_space, tenv.action_space, cfg=cfg).to(cfg.device)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(cfg, obs_dim=obs_dim, act_dim=act_dim)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(utils.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
    sum_count = sum(var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        alpha = cfg.alpha if not cfg.learned_alpha else torch.exp(ac.log_alpha).detach()
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + cfg.gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        alpha = cfg.alpha if not cfg.learned_alpha else torch.exp(ac.log_alpha).detach()
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        if cfg.learned_alpha:
            loss_alpha = (-1.0 * (torch.exp(ac.log_alpha) * (logp_pi + ac.target_entropy).detach())).mean()
        else:
            loss_alpha = 0.0

        return loss_pi, loss_alpha, pi_info


    # Set up optimizers for policy and q-function
    q_optimizer = Adam(q_params, lr=cfg.vf_lr)
    pi_optimizer = Adam(ac.pi.parameters(), lr=cfg.pi_lr)
    if cfg.learned_alpha:
        alpha_optimizer = Adam([ac.log_alpha], lr=cfg.alpha_lr)

    def update(data):
        # TODO: optimize this by not requiring the items right away.
        # i think this might be blockin for pytorch to finish some computations

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger['LossQ'] += [loss_q.detach().cpu()]
        for key in q_info:
            logger[key] += [q_info[key]]

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, loss_alpha, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()
        # and optionally the alpha
        if cfg.learned_alpha:
            alpha_optimizer.zero_grad()
            loss_alpha.backward()
            alpha_optimizer.step()
            logger['LossAlpha'] += [loss_alpha.detach().cpu()]
            logger['Alpha'] += [torch.exp(ac.log_alpha.detach().cpu())]

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger['LossPi'] += [loss_pi.item()]
        for key in pi_info:
            logger[key] += [pi_info[key]]

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(cfg.polyak)
                p_targ.data.add_((1 - cfg.polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32).to(cfg.device), deterministic)

    def test_agent(video=False):
        frames = []
        for j in range(cfg.num_test_episodes):
            o, d, ep_ret, ep_len = tenv.reset(), False, 0, 0
            while not(d or (ep_len == cfg.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = tenv.step(get_action(o[None], True)[0])
                ep_ret += r
                ep_len += 1
                if video and j == 0:
                    frame = deepcopy(tenv.render(mode='rgb_array'))
                    frames.append(PIL.Image.fromarray(frame).resize([128,128]))
                    if d:
                        tenv.step(tenv.action_space.sample())
                        frame = deepcopy(tenv.render(mode='rgb_array'))
                        frames.append(PIL.Image.fromarray(frame).resize([128,128]))
            logger['TestEpRet'] += [ep_ret]
            logger['TestEpLen'] += [ep_len]

        if len(frames) != 0:
            vid = np.stack(frames)
            vid_tensor = vid.transpose(0,3,1,2)[None]
            writer.add_video('rollout', vid_tensor, epoch, fps=60)
            frames = []
            writer.flush()
            print('wrote video')
    test_agent()

    # Prepare for interaction with environment
    total_steps = cfg.steps_per_epoch * cfg.epochs
    epoch_time = start_time = time.time()
    o, ep_ret, ep_len = venv.reset(), np.zeros(cfg.num_envs), np.zeros(cfg.num_envs)
    # Main loop: collect experience in venv and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > cfg.start_steps:
            a = get_action(o)
        else:
            a = venv.action_space.sample()

        # Step the venv
        o2, r, d, _ = venv.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        time_horizon = ep_len==cfg.max_ep_len
        d[ep_len==cfg.max_ep_len] = False

        # Store experience to replay buffer
        replay_buffer.store_n({'obs': o, 'act': a, 'rew': r, 'obs2': o2, 'done': d})

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        done = np.logical_or(d, ep_len == cfg.max_ep_len)
        for idx in np.nonzero(done)[0]:
            logger['EpRet'] += [ep_ret[idx]]
            logger['EpLen'] += [ep_len[idx]]
            ep_ret[idx] = 0
            ep_len[idx] = 0

        # Update handling
        if t >= cfg.update_after and t % cfg.update_every == 0:
            for j in range(int(cfg.update_every*1.5)):
                batch = replay_buffer.sample_batch(cfg.bs)
                update(data=batch)

        # End of epoch handling
        if (t+1) % cfg.steps_per_epoch == 0:
            epoch = (t+1) // cfg.steps_per_epoch

            # Save model
            if (epoch % cfg.save_freq == 0) or (epoch == cfg.epochs):
                pass

            # Test the performance of the deterministic version of the agent.
            if epoch % 5 == 0:
                test_agent(video=epoch%5==0)

            # Log info about epoch
            print('='*30)
            print('Epoch', epoch)
            logger['var_count'] = [sum_count]
            for key in logger:
                val = np.mean(logger[key])
                writer.add_scalar(key, val, epoch)
                print(key, val)
            if cfg.net == 'router':
                writer.add_image('Pi/i_route', ac.pi.net.get_iperm()[None], epoch)
                writer.add_image('Pi/o_route', ac.pi.net.get_operm()[None], epoch)
                writer.add_image('Q1/i_route', ac.q1.get_iperm()[None], epoch)
                writer.add_image('Q2/i_route', ac.q2.get_iperm()[None], epoch)
            writer.flush()
            print('TotalEnvInteracts', t*cfg.num_envs)
            print('Time', time.time()-start_time)
            print('dt', time.time()-epoch_time)
            print(logpath)
            print(cfg.full_cmd)
            print('='*30)
            logger = defaultdict(lambda: [])
            epoch_time = time.time()


if __name__ == '__main__':
    parser = define_cfg()
    #parser.set_defaults(**{'logdir': 'logs/sac/'})
    cfg = parser.parse_args()
    sac(cfg)