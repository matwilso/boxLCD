import argparse
import copy
import itertools
import os
import pathlib
import time
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
import yaml
from model import GPT
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import parse_args

from boxLCD import env_map, envs
from boxLCD.utils import A, AttrDict, args_type


class Trainer:
    def __init__(self, G):
        self.G = G
        self.env = env_map[G.env](G)
        self.act_dim = self.env.action_space.shape[0]
        self.G.lcd_w = int(self.G.lcd_base * self.G.wh_ratio)
        self.G.lcd_h = self.G.lcd_base
        self.model = GPT(self.act_dim, G)
        self.optimizer = Adam(self.model.parameters(), lr=G.lr)
        self.G.num_vars = self.num_vars = utils.count_vars(self.model)
        print('num vars', self.num_vars)
        self.train_ds, self.test_ds = utils.load_ds(G)
        self.writer = SummaryWriter(G.logdir)
        self.logger = utils.dump_logger({}, self.writer, 0, G)

    def train_epoch(self, i):
        """run a single epoch of training over the data"""
        self.optimizer.zero_grad()
        for batch in self.train_ds:
            batch = {key: val.to(self.G.device) for key, val in batch.items()}
            self.optimizer.zero_grad()
            loss = self.model.loss(batch)
            loss.backward()
            self.optimizer.step()
            self.logger['loss'] += [loss.detach().cpu()]

    def sample(self, i):
        # TODO: prompt to a specific point and sample from there. to compare against ground truth.
        N = 5
        action = (torch.rand(N, self.G.ep_len, self.env.action_space.shape[0]) * 2 - 1).to(
            self.G.device
        )
        sample, sample_loss = self.model.sample(N, action=action)
        self.logger['sample_loss'] += [sample_loss]
        lcd = sample['lcd']
        lcd = (
            lcd.cpu().detach().repeat_interleave(4, -1).repeat_interleave(4, -2)[:, 1:]
        )
        self.writer.add_video(
            'unprompted_samples', utils.force_shape(lcd), i, fps=self.G.fps
        )
        # EVAL
        if (
            len(self.env.world_def.robots) == 0
        ):  # if we are just dropping the object, always use the same setup
            if 'BoxOrCircle' == self.G.env:
                reset_states = np.c_[
                    np.ones(N), np.zeros(N), np.linspace(-0.8, 0.8, N), 0.5 * np.ones(N)
                ]
            else:
                reset_states = np.c_[
                    np.random.uniform(-1, 1, N),
                    np.random.uniform(-1, 1, N),
                    np.linspace(-0.8, 0.8, N),
                    0.5 * np.ones(N),
                ]
        else:
            reset_states = [None] * N
        obses = {
            key: [[] for ii in range(N)] for key in self.env.observation_space.spaces
        }
        acts = [[] for ii in range(N)]
        for ii in range(N):
            for key, val in self.env.reset(reset_states[ii]).items():
                obses[key][ii] += [val]
            for _ in range(self.G.ep_len - 1):
                act = self.env.action_space.sample()
                obs = self.env.step(act)[0]
                for key, val in obs.items():
                    obses[key][ii] += [val]
                acts[ii] += [act]
            acts[ii] += [np.zeros_like(act)]
        obses = {key: np.array(val) for key, val in obses.items()}
        action = np.array(acts)
        action = torch.as_tensor(action, dtype=torch.float32).to(self.G.device)
        prompts = {
            key: torch.as_tensor(1.0 * val[:, :10]).to(self.G.device)
            for key, val in obses.items()
        }
        prompted_samples, prompt_loss = self.model.sample(
            N, action=action, prompts=prompts
        )
        self.logger['prompt_sample_loss'] += [prompt_loss]
        real_lcd = obses['lcd'][:, :, None]
        lcd_psamp = prompted_samples['lcd']
        lcd_psamp = lcd_psamp.cpu().detach().numpy()
        error = (lcd_psamp - real_lcd + 1.0) / 2.0
        blank = np.zeros_like(real_lcd)[..., :1, :]
        out = np.concatenate([real_lcd, blank, lcd_psamp, blank, error], 3)
        out = out.repeat(4, -1).repeat(4, -2)
        self.writer.add_video('prompted_lcd', utils.force_shape(out), i, fps=self.G.fps)

    def test(self, i):
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_ds:
                batch = {key: val.to(self.G.device) for key, val in batch.items()}
                loss = self.model.loss(batch)
                self.logger['test_loss'] += [loss.mean().detach().cpu()]
        sample_start = time.time()
        if i % self.C.log_n == 0:
            self.sample(i)
        self.logger['dt/sample'] = [time.time() - sample_start]
        self.logger['num_vars'] = self.num_vars
        self.logger = utils.dump_logger(self.logger, self.writer, i, self.G)
        self.writer.flush()
        self.model.train()

    def save(self, i=0):
        path = self.C.logdir / f'model.pt'
        print("SAVED MODEL", path)
        torch.save(self.model.state_dict(), path)

    def run(self):
        for i in itertools.count():
            self.train_epoch(i)
            self.test(i)
            if i >= self.G.num_epochs:
                break
        self.save(i)


if __name__ == '__main__':
    G = parse_args()
    trainer = Trainer(G)
    trainer.run()
