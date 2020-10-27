
import uuid
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
import utils
from nets.sacnets import ActorCritic
from functools import partial
from tensorflow import nest
from data import records

# TODO: add support for saving weights to file and loading

class Trainer:
    def __init__(self, cfg, make_env):
        self.cfg = cfg
        print(self.cfg.full_cmd)
        seed = self.cfg.seed
        # Set up logger and save configuration
        self.logger = defaultdict(lambda: [])
        timestamp = datetime.now().strftime('%Y%m%dT-%H-%M-%S')
        self.logpath = self.cfg.logdir/f'{self.cfg.env}/{self.cfg.exp_name}-{timestamp}'
        self.writer = SummaryWriter(self.logpath)
        self.barrel_path = self.cfg.barrel_path if self.cfg.barrel_path != '' else self.logpath / 'barrels'

        print(self.logpath)
        print(cfg.full_cmd)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.venv = AsyncVectorEnv([make_env(self.cfg, i) for i in range(self.cfg.num_envs)]) # vector env
        self.tenv = make_env(self.cfg, seed)() # test env
        self.state_shape = self.tenv.observation_space['state'].shape
        self.image_shape = self.tenv.observation_space['image'].shape
        self.act_n = self.tenv.action_space.shape[0]

        ## Create actor-critic module and target networks
        #self.ac = ActorCritic(self.tenv, self.tenv.observation_space, self.tenv.action_space, cfg=self.cfg).to(cfg.device)
        #self.ac_targ = deepcopy(self.ac)

        ## Freeze target networks with respect to optimizers (only update via polyak averaging)
        #for p in self.ac_targ.parameters():
        #    p.requires_grad = False

        ## List of parameters for both Q-networks (save this for convenience)
        #self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        ## Count variables (protip: try to get a feel for how different size networks behave!)
        #var_counts = tuple(utils.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        #print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        #self.sum_count = sum(var_counts)

        ## Set up optimizers for policy and q-function
        #self.q_optimizer = Adam(self.q_params, lr=self.cfg.vf_lr)
        #self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.cfg.pi_lr)
        #if self.cfg.learned_alpha:
        #    self.alpha_optimizer = Adam([self.ac.log_alpha], lr=self.cfg.alpha_lr)

        #self.test_agent()

        self.data_iter = None

    @property
    def can_sample(self):
        return len(list(self.barrel_path.glob('*.tfrecord'))) != 0

    def refresh_dataset(self):
        self.data_iter = records.make_dataset(self.barrel_path, self.state_shape, self.image_shape, self.act_n, self.cfg)

    def sample_batch(self, refresh_dataset=False):
        """return dict(str: arr) where arr is shape (BS, BPTT_N, ...)"""
        assert self.can_sample
        if self.data_iter is None or refresh_dataset:
            self.refresh_dataset()
        batch = next(self.data_iter)
        batch = nest.map_structure(lambda x: jnp.array(x), batch)
        return batch

    def test_agent(self, video=False):
        frames = []
        for j in range(self.cfg.num_test_episodes):
            o, d, ep_ret, ep_len = self.tenv.reset(), False, 0, 0
            while not(d or (ep_len == self.cfg.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = self.tenv.step(self.get_action(o[None], True)[0])
                ep_ret += r
                ep_len += 1
                if video and j == 0:
                    frame = deepcopy(self.tenv.render(mode='rgb_array'))
                    frames.append(PIL.Image.fromarray(frame).resize([128,128]))
                    if d:
                        self.tenv.step(self.tenv.action_space.sample())
                        frame = deepcopy(self.tenv.render(mode='rgb_array'))
                        frames.append(PIL.Image.fromarray(frame).resize([128,128]))
            self.logger['TestEpRet'] += [ep_ret]
            self.logger['TestEpLen'] += [ep_len]

        if len(frames) != 0:
            vid = np.stack(frames)
            vid_tensor = vid.transpose(0,3,1,2)[None]
            self.writer.add_video('rollout', vid_tensor, self.t, fps=60)
            frames = []
            self.writer.flush()
            print('wrote video')
    
    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.cfg.device), deterministic)

    def collect_episode(self, eplen, num_ep):
        num = num_ep // self.cfg.num_envs
        tot = defaultdict(lambda: [])
        for n in range(num):
            pack = defaultdict(lambda: [])
            o = self.venv.reset()
            pack['state'] += [o['state']]
            pack['image'] += [o['image']/255.0]
            for t in range(eplen):
                a = self.venv.action_space.sample()
                o2, r, d, _ = self.venv.step(a)
                pack['act'] += [a]
                pack['rew'] += [r]
                pack['state'] += [o2['state']]
                pack['image'] += [o2['image']/255.0]
                o = o2
            for key in pack:
                tot[key] += [pack[key]]
        for key in tot:
            tk = np.array(tot[key])
            tk = tk.swapaxes(2,1)
            tk = tk.reshape((-1,)+tk.shape[2:])
            tot[key] = tk

        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        identifier = str(uuid.uuid4().hex)
        self.barrel_path.mkdir(parents=True, exist_ok=True)
        filename =  self.barrel_path / f'{timestamp}-{identifier}{self.cfg.exp_name}-{num_ep}-{eplen}.tfrecord'
        records.write_barrel(filename, tot)