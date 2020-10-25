import torch
from algo.base import Trainer
import numpy as np

class Collect(Trainer):
    def __init__(self, cfg, make_env):
        super().__init__(cfg, make_env)

    def run(self):
        self.collect_episode(100, 100)
        #total_steps = self.cfg.steps_per_epoch * self.cfg.epochs
        #o = self.venv.reset()
        #for self.t in range(total_steps):
        #    a = self.venv.action_space.sample()
        #    o2, r, d, _ = self.venv.step(a)
        #    # handle done
        #    self.replay_buffer.store_n({'obs': o, 'act': a, 'rew': r, 'obs2': o2})
        #    o = o2