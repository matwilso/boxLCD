import pathlib
import subprocess
import sys

import gym

import boxLCD.utils
from boxLCD import ENV_DG, env_map
from boxLCD.utils import args_type
from research import wrappers


def env_fn(G, seed=None):
    def _make():
        if G.env in env_map:
            env = env_map[G.env](G)
            env.seed(seed)
            if G.goals:
                if 'Cube' not in G.env:
                    env = wrappers.BodyGoalEnv(env, G)
                else:
                    env = wrappers.CubeGoalEnv(env, G)
        else:
            env = gym.make(G.env)
            env = wrappers.WrappedGym(env, G)
            env.seed(seed)
        return env

    return _make


def config():
    G = boxLCD.utils.AttrDict()
    # BASICS
    G.logdir = pathlib.Path('./logs/trash')
    G.weightdir = pathlib.Path('.')
    G.buffdir = pathlib.Path('.')
    G.datadir = pathlib.Path('.')
    G.arbiterdir = pathlib.Path('.')
    G.device = 'cuda'  # 'cuda', 'cpu'
    G.mode = 'train'
    G.model = 'BVAE'
    G.datamode = 'video'
    G.ipython_mode = 0

    # G.data_mode = 'image'
    G.amp = 0
    G.total_itr = int(1e9)
    G.log_n = int(1e4)
    G.save_n = 5
    G.refresh_data = 0

    G.decode = 'multi'
    G.conv_io = 0
    G.train_barrels = -1  # -1 means all. any other number is how many to use
    G.test_barrels = 1
    G.grad_clip = 10.0

    G.bs = 64
    G.lr = 1e-4
    G.n_layer = 2
    G.n_head = 4
    G.n_embed = 128
    G.hidden_size = 128
    G.nfilter = 64
    G.vidstack = -1
    G.stacks_per_block = 32

    G.vqD = 128
    G.vqK = 128
    G.beta = 0.25
    G.entropy_bonus = 0.0

    G.min_std = 1e-4
    G.data_frac = 1.0
    G.vanished = 1
    G.num_envs = 8

    G.mdn_k = 5
    G.dist_delta = 0
    G.sample_sample = 0
    G.skip_train = 0

    G.phase = 1
    G.window = 50
    G.seed = 0
    G.end2end = 0

    G.video_n = 8
    G.prompt_n = 8

    G.env = 'Dropbox'
    G.goals = 0
    G.preproc = 0
    G.state_rew = 1
    G.rew_scale = 1.0
    G.free_nats = 3.0
    G.kl_scale = 1.0
    G.autoreset = 0

    G.make_video = 0
    G.data_workers = 12

    # extra info that we set here for convenience and don't modify
    G.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
    G.commit = (
        subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        .strip()
        .decode('utf-8')
    )

    # values set by the code
    G.num_vars = 0

    pastKeys = list(G.keys())
    for key, val in ENV_DG.items():
        assert key not in pastKeys, f'make sure you are not duplicating keys {key}'
        G[key] = val

    return G
