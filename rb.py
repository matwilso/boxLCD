from collections import defaultdict
import torch
import numpy as np
import utils
from tensorflow import nest

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, config, obs_dim, act_dim):
        self.config = config
        size = config.replay_size
        self.bufs = {}
        self.bufs['obs'] = np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        self.bufs['obs2'] = np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        self.bufs['act'] = np.zeros(utils.combined_shape(size, act_dim), dtype=np.float32)
        self.bufs['rew'] = np.zeros(size, dtype=np.float32)
        self.bufs['done'] = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store_n(self, ntrans):
        shape = self.config.num_envs
        end_idx = self.ptr + shape
        if end_idx <= self.max_size: # normal operation
            for key in self.bufs:
                self.bufs[key][self.ptr:end_idx] = ntrans[key]
            self.ptr = (self.ptr + shape) % self.max_size
        else: # handle wrap around
            overflow = (end_idx - self.max_size)
            top_off = shape - overflow
            for key in self.bufs:
                self.bufs[key][self.ptr:self.ptr+top_off] = ntrans[key][:top_off] # top off the last end of the array
                self.bufs[key][:overflow] = ntrans[key][top_off:]  # start over at beginning
            self.ptr = overflow
        self.size = min(self.size+shape, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = nest.map_structure(lambda x: x[idxs], self.bufs)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.config.device) for k,v in batch.items()}