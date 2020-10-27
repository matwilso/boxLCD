import torch
from algo.base import Trainer
import numpy as np

class Collect(Trainer):
    def __init__(self, cfg, make_env):
        super().__init__(cfg, make_env)

    def run(self):
        num_files = self.cfg.replay_size // (self.cfg.ep_len * self.cfg.num_eps)
        print(num_files)
        for _ in range(num_files):
            self.collect_episode(self.cfg.ep_len, self.cfg.num_eps)
        #self.refresh_dataset()
        #batch = next(self.data_iter)