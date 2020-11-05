import argparse
import sys
import pathlib
import utils

"""
Command line args.
"""

def args_type(default):
    if isinstance(default, bool):
        return lambda x: bool(['False', 'True'].index(x))
    if isinstance(default, int):
        return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    return type(default)

def make_env(cfg, seed):
    def _make():
        env = env_fn(cfg, seed)
        return env
    return _make

def env_fn(cfg, seed=None):
    """function for creating an env with the cfg and seed"""
    from envs.wrappers import PixelEnv, NormalEnv
    if cfg.env == 'fishes':
        from envs.fishes import Fishes
        env = Fishes(cfg)
    elif cfg.env == 'box':
        from envs.box import Box
        env = Box(cfg)
    elif cfg.env == 'llc':
        import gym
        env = gym.make('LunarLanderContinuous-v2')
    env.seed(seed)
    if cfg.use_image:
        env = PixelEnv(env)
    else:
        env = NormalEnv(env)
    return env

def define_cfg():
    cfg = utils.AttrDict() # dictionary that can grab items with dot notation.


    # BASICS
    cfg.logdir = pathlib.Path('logs/')
    cfg.barrel_path = ''
    cfg.device = 'cuda' # 'cuda', 'cpu'
    cfg.env = 'box'
    cfg.num_envs = 10
    cfg.seed = 0
    cfg.epochs = 100
    cfg.steps_per_epoch = 1000
    cfg.save_freq = 10
    cfg.name = ''
    cfg.pi_lr = 3e-4
    cfg.vf_lr = 3e-4
    cfg.dyn_lr = 1e-3
    cfg.lds_lr = 3e-4
    cfg.alpha_lr = 1e-3 # for SAC w/ learned alpha
    cfg.net = 'mlp' # 'mlp', 'split'
    cfg.bs = 50
    cfg.bl = 50
    cfg.horizon = 15


    cfg.reward_mode = 'explore'

    cfg.num_ens = 5
    cfg.act_ent_weight = 3e-3
    cfg.rew_weight = 1.0

    cfg.polyak = 0.995

    # ARCHITECTURE
    cfg.split_share = 0  # whether or not to share weights in the split network

    # 
    cfg.ep_len = 150
    cfg.num_eps = 100

    cfg.mode = 'dream'

    cfg.stoch = 30
    cfg.deter = 200
    cfg.hidden = 200
    cfg.kl_scale = 1.0
    cfg.log_n = 1000
    cfg.env_size = 64

    cfg.obs_stats = 0

    cfg.use_image = 0

    # ----- RL -----
    cfg.gamma = 0.99
    cfg.lam = 0.95
    # SAC
    cfg.learned_alpha = 1
    cfg.alpha = 0.1 # for SAC w/o learned alpha
    cfg.polyak = 0.995
    cfg.num_test_episodes = 2
    cfg.replay_size = int(1e6)
    cfg.update_every = 40
    cfg.start_steps = 1000
    cfg.update_after = 1000
    # ------------------

    # extra info that we set here for convenience and don't modify 
    cfg.full_cmd = 'python ' + ' '.join(sys.argv)  # full command that was called
    cfg.clipboard = 0

    parser = argparse.ArgumentParser()
    for key, value in cfg.items():
        parser.add_argument(f'--{key}', type=args_type(value), default=value)
    return parser