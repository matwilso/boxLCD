import pathlib
import yaml
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
nms = nest.map_structure

# TODO: add support for saving weights to file and loading

class Trainer:
    def __init__(self, cfg, make_env):
        self.cfg = cfg
        assert cfg.ep_len % cfg.bl == 0 
        seed = self.cfg.seed
        # Set up logger and save configuration
        self.logger = defaultdict(lambda: [])
        timestamp = datetime.now().strftime('%Y%m%dT-%H-%M-%S')
        self.logpath = self.cfg.logdir/f'{self.cfg.env}/{self.cfg.name}-{timestamp}'
        if cfg.clipboard:
            import pyperclip
            pyperclip.copy(str(self.logpath))

        self.writer = SummaryWriter(self.logpath)
        self.barrel_path = pathlib.Path(self.cfg.barrel_path) if self.cfg.barrel_path != '' else self.logpath / 'barrels'
        if self.barrel_path.name != 'barrels': self.barrel_path = self.barrel_path / 'barrels'

        print(self.logpath)
        print(cfg.full_cmd)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.venv = AsyncVectorEnv([make_env(self.cfg, i) for i in range(self.cfg.num_envs)]) # vector env
        self.tenv = make_env(self.cfg, seed)() # test env
        self.state_shape = self.tenv.observation_space['state'].shape
        if self.cfg.use_image:
            self.image_shape = self.tenv.observation_space['image'].shape
        else:
            self.image_shape = (64, self.cfg.env_wh_ratio*64, 3)
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
        with open(self.logpath/'flags.yaml', 'w') as f:
            yaml.dump(self.cfg.__dict__, f)


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
        batch = nms(lambda x: jnp.array(x), batch)
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

    def logger_dump(self):
        print('='*30)
        print('t', self.t)
        for key in self.logger:
            x = np.mean(self.logger[key])
            self.writer.add_scalar(key, x, self.t)
            print(key, x)
        dt = time.time()-self.dt_time
        self.writer.add_scalar('dt', dt, self.t)
        self.writer.flush()
        print('dt', dt)
        print('total time', time.time()-self.start_time)
        print(self.logpath)
        print(self.cfg.full_cmd)
        print('='*30)
        self.dt_time = time.time()

    def get_batch(self):
        bt = time.time()
        batch = next(self.data_iter)
        if self.cfg.use_image:
            batch = nest.map_structure(lambda x: torch.tensor(x).to(self.cfg.device), batch)
            batch['image'] = (batch['image'].permute([0, 1, -1, 2, 3]) / 255.0) - 0.5
            #batch = nest.map_structure(lambda x: x.flatten(0,1), batch)
        else:
            if 'image' in batch: batch.pop('image')
            # TODO: make this an EMA with var
            if self.cfg.obs_stats:
                self.obs_rms.update(batch['state'].reshape([self.cfg.bs*self.cfg.bl, -1]))
                batch['state'] = (batch['state'] - self.obs_rms.mean) / (1.0*self.obs_rms.var**0.5)
                batch['std'] = self.obs_rms.var**0.5
            batch = nest.map_structure(lambda x: torch.tensor(x).float().to(self.cfg.device), batch)
        self.logger['dt/batch'] += [time.time() - bt]
        return batch

    def collect_episode(self, eplen, num_ep, mode='random'):
        num = num_ep // self.cfg.num_envs
        tot = defaultdict(lambda: [])
        for n in range(num):
            o = self.venv.reset()
            agent_state = None
            pack = defaultdict(lambda: [])
            for t in range(eplen):
                if mode == 'random':
                    a = self.venv.action_space.sample()
                else:
                    a, agent_state = self.policy(o, agent_state, training=True)
                    a = np.array(a.detach().cpu())
                o2, r, d, _ = self.venv.step(a)
                pack['state'] += [o['state']]
                if self.cfg.use_image:
                    pack['image'] += [o['image']/255.0]
                pack['act'] += [a]
                pack['rew'] += [r]
                o = o2
            for key in pack:
                tot[key] += [pack[key]]
        for key in tot:
            tk = np.array(tot[key])
            tk = tk.swapaxes(2,1)
            tk = tk.reshape((-1,)+tk.shape[2:])
            tot[key] = tk

        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        self.logger['reward/collect'] += [np.mean(pack['rew'])]
        identifier = str(uuid.uuid4().hex)
        self.barrel_path.mkdir(parents=True, exist_ok=True)
        filename =  self.barrel_path / f'{timestamp}-{identifier}{self.cfg.name}-{num_ep}-{eplen}'
        tot = nms(lambda x: x.reshape([num_ep, x.shape[1]//self.cfg.bl, self.cfg.bl, -1]), tot)
        for i in range(tot['state'].shape[1]):
            records.write_barrel(filename.with_suffix(f'.{i}.tfrecord'), nms(lambda x: x[:,i], tot), self.cfg)