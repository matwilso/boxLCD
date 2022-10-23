import itertools
import time
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from boxLCD.utils import A
from research import utils
from research.wrappers import AsyncVectorEnv

BARREL_SIZE = int(1e3)
from einops import rearrange

# from jax.tree_util import tree_map, tree_map

"""
Crack yourself open a fresh barrel of data
"""


def collect(env_fn, G):
    collect_start = time.time()
    G.logdir.mkdir(parents=True, exist_ok=True)
    utils.dump_logger({}, None, 0, G)
    env = env_fn(G)()
    venv = AsyncVectorEnv([env_fn(G) for _ in range(G.num_envs)], G=G)
    assert (
        BARREL_SIZE % G.num_envs == 0
    ), f'barrel size must be divisible by number of envs you use {BARREL_SIZE} % {G.num_envs} != 0'
    assert (
        G.train_barrels != -1 and G.test_barrels != -1
    ), f'must set the number of barrels you want to fill. G.train_barrels=={G.train_barrels}'
    fill_barrels(env, venv, G.test_barrels, 'test', G)
    fill_barrels(env, venv, G.train_barrels, 'train', G)
    print('TOTAL COLLECT TIME', time.time() - collect_start)


def fill_barrels(env, venv, num_barrels, prefix, G):
    """Create files with:
    BARREL_SIZE x EP_LEN x *STATE_DIMS

    o1,a1 --> o2
    Meaning that the last action doesn't matter
    """
    BARS = BARREL_SIZE // G.num_envs
    logdir = G.logdir / prefix
    logdir.mkdir(parents=True, exist_ok=True)
    total_bar = tqdm(total=num_barrels)
    barrel_bar = tqdm(total=BARS)
    total_bar.set_description(f'TOTAL PROGRESS (FPS=N/A)')
    for ti in range(num_barrels):
        obses = {
            key: np.zeros([BARS, G.num_envs, G.ep_len, *val.shape], dtype=val.dtype)
            for key, val in env.observation_space.spaces.items()
        }
        acts = np.zeros([BARS, G.num_envs, G.ep_len, env.action_space.shape[0]])
        barrel_bar.reset()
        for bi in range(BARS):
            start = time.time()
            obs = venv.reset(np.arange(G.num_envs))
            for j in range(G.ep_len):
                act = venv.action_space.sample()
                for key in obses:
                    obses[key][bi, :, j] = obs[key]
                acts[bi, :, j] = np.stack(act)
                obs, rew, done, info = venv.step(act)
                # plt.imshow(obs['lcd']);plt.show()
                # venv.render()
                # plt.imshow(1.0*venv.lcd_render()); plt.show()
            barrel_bar.update(1)
            fps = G.ep_len / (time.time() - start)
            barrel_bar.set_description(f'current barrel')
            # barrel_bar.set_description(f'fps: {} | current barrel')
        if (G.logdir / 'pause.marker').exists():
            import ipdb

            ipdb.set_trace()

        obses = {key: utils.flatten_first(val) for key, val in obses.items()}
        acts = utils.flatten_first(acts)
        assert obses['proprio'].ndim == 3
        assert acts.ndim == 3
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        data = np.savez_compressed(
            logdir / f'{timestamp}-{G.ep_len}.barrel', action=acts, **obses
        )
        total_bar.update(1)
        total_bar.set_description(f'TOTAL PROGRESS (FPS={fps})')


def fill_barrels_slow(env, num_barrels, prefix, G):
    """Create files with:
    BARREL_SIZE x EP_LEN x *STATE_DIMS

    o1,a1 --> o2
    Meaning that the last action doesn't matter
    """
    import ipdb

    ipdb.set_trace()
    logdir = G.logdir / prefix
    logdir.mkdir(parents=True, exist_ok=True)
    total_bar = tqdm(total=num_barrels)
    barrel_bar = tqdm(total=BARREL_SIZE)
    total_bar.set_description(f'TOTAL PROGRESS (FPS=N/A)')
    for ti in range(num_barrels):
        obses = {
            key: np.zeros([BARREL_SIZE, G.ep_len, *val.shape], dtype=val.dtype)
            for key, val in env.observation_space.spaces.items()
        }
        acts = np.zeros([BARREL_SIZE, G.ep_len, env.action_space.shape[0]])
        barrel_bar.reset()
        for bi in range(BARREL_SIZE):
            start = time.time()
            obs = env.reset()
            for j in range(G.ep_len):
                act = env.action_space.sample()
                for key in obses:
                    obses[key][bi, j] = obs[key]
                acts[bi, j] = act
                obs, rew, done, info = env.step(act)
                # plt.imshow(obs['lcd']);plt.show()
                # env.render()
                # plt.imshow(1.0*env.lcd_render()); plt.show()
            barrel_bar.update(1)
            fps = G.ep_len / (time.time() - start)
            barrel_bar.set_description(f'current barrel')
            # barrel_bar.set_description(f'fps: {} | current barrel')
        if (G.logdir / 'pause.marker').exists():
            import ipdb

            ipdb.set_trace()

        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        data = np.savez_compressed(
            logdir / f'{timestamp}-{G.ep_len}.barrel', action=acts, **obses
        )
        total_bar.update(1)
        total_bar.set_description(f'TOTAL PROGRESS (FPS={fps})')


class RolloutDataset(IterableDataset):
    def __init__(self, barrel_path, window=int(1e9), infinite=True, refresh_data=False, max_barrels=None):
        super().__init__()
        self.window = window
        self.infinite = infinite
        self.barrel_path = barrel_path
        self.refresh_data = refresh_data
        self.max_barrels = max_barrels
        self._refresh()

    def _refresh(self):
        """recheck the directory for new barrels"""
        self.barrel_files = list(self.barrel_path.glob('*.barrel.npz'))[:self.max_barrels]
        self.nbarrels = len(self.barrel_files)
        assert self.nbarrels > 0, 'didnt find any barrels at datadir'

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.id + round(time.time()))

        for ct in itertools.count():
            if self.infinite:
                curr_file = self.barrel_files[np.random.randint(self.nbarrels)]
                if self.refresh_data and ct % 10 == 0:
                    self._refresh()
            else:
                curr_file = self.barrel_files[ct]

            curr_barrel = np.load(curr_file, allow_pickle=True)
            elems = {
                key: torch.as_tensor(curr_barrel[key], dtype=torch.float32)
                for key in curr_barrel.keys()
            }
            ep_len = elems['lcd'].shape[1]
            # elems = {key: torch.as_tensor(np.c_[np.zeros_like(curr_barrel[key])[:self.window], curr_barrel[key]], dtype=torch.float32) for key in curr_barrel.keys()}
            pad = self.window - 1
            # elems = {key: torch.cat([torch.zeros_like(val)[:,:pad], val, torch.zeros_like(val)[:,:pad]], axis=1) for key, val in elems.items()}
            lcd = elems['lcd']

            idxs = np.arange(BARREL_SIZE)
            np.random.shuffle(idxs)
            max_start = elems['lcd'].shape[1] - self.window

            # TODO: pad the elems with some zero elements, then we can always use the same logic
            # TODO: add something like timestamps and clip id to the observation. unique id and then time in that id.

            for idx in idxs:
                if max_start > 0:
                    start = np.random.randint(0, max_start)
                    elem = {
                        key: torch.as_tensor(
                            val[idx, start : start + self.window], dtype=torch.float32
                        )
                        for key, val in elems.items()
                    }
                else:
                    elem = {
                        key: torch.as_tensor(val[idx], dtype=torch.float32)
                        for key, val in elems.items()
                    }

                #  start = np.random.randint(0, max_start)
                # start = np.random.randint(0, ep_len)
                # elem = {key: torch.as_tensor(val[idx, start:start+self.window], dtype=torch.float32) for key, val in elems.items()}
                elem['lcd'] = rearrange(elem['lcd'], 't h w c -> c t h w', c=3)
                elem['lcd'] /= 255.0
                # if (elem['lcd'] == 0.0).all(dim=0).all(dim=-1).all(dim=-1).any():
                #  import ipdb; ipdb.set_trace()
                assert elem['lcd'].max() <= 1.0 and elem['lcd'].min() >= 0.0
                yield elem
            curr_barrel.close()
            if ct >= self.nbarrels - 1 and not self.infinite:
                break


def load_ds(G):
    test_dset = RolloutDataset(G.datadir / 'test', G.window, infinite=False, max_barrels=1)
    test_loader = DataLoader(
        test_dset,
        batch_size=G.bs,
        #batch_size=8,
        pin_memory=G.device == 'cuda',
        num_workers=G.data_workers,
        drop_last=True,
    )

    train_dset = RolloutDataset(
        G.datadir / 'train', G.window, refresh_data=G.refresh_data
    )
    train_loader = DataLoader(
        train_dset,
        batch_size=G.bs,
        pin_memory=G.device == 'cuda',
        num_workers=G.data_workers,
        drop_last=True,
    )

    train_loader.nbarrels = train_dset.nbarrels
    test_loader.nbarrels = test_dset.nbarrels
    print(f"Number of train barrels {train_dset.nbarrels}")
    print(f"Number of test barrels {test_dset.nbarrels}")
    return train_loader, test_loader
