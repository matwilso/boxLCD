import io
import uuid
from datetime import datetime
import pathlib
from threading import Thread, Lock
from collections import defaultdict
import copy
from tensorflow import nest
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as onp
import jax.numpy as jnp
import utils
from data import records
A = utils.A

land = onp.logical_and
lor = onp.logical_or
lnot = onp.logical_not

KEYS = ['obs', 'act', 'rew']
class Roller:
    def __init__(self, obs_n, act_n, test_env, example_state, phi_size, FLAGS, algo=None, act_hl_n=None):
        self.FLAGS = FLAGS
        self.obs_n = obs_n
        self.act_n = act_n
        max_size = self.FLAGS['replay_size']
        self.algo = algo
        self.keys = KEYS
        self.test_env = test_env
        self.ep_id = onp.arange(FLAGS['num_envs'])
        # ep buffers are where the rollout thread puts stuff during an episode.
        self.ep_buffers = [{key: [] for key in self.keys}
            for _ in range(FLAGS['num_envs'])]
        # the loading dock holds all episodes yet to be processed.
        self.loading_dock = []
        self.barrel_path = self.FLAGS['barrel_path']
        assert self.FLAGS['num_per_barrel'] % self.FLAGS['env:timeout'] == 0
        self.data_iter = None

    @property
    def can_sample(self):
        return len(list(self.barrel_path.glob('*.tfrecord'))) != 0

    def refresh_dataset(self):
        self.data_iter = records.make_dataset(self.obs_n, self.act_n, self.FLAGS)

    def sample_batch(self, refresh_dataset=False):
        """return dict(str: arr) where arr is shape (BS, BPTT_N, ...)"""
        assert self.can_sample
        if self.data_iter is None or refresh_dataset:
            self.refresh_dataset()
        batch = next(self.data_iter)
        batch = nest.map_structure(lambda x: jnp.array(x), batch)
        return batch
