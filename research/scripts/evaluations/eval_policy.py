import argparse
from collections import defaultdict
from shutil import ignore_patterns

import matplotlib.pyplot as plt
import numpy as np
import torch
from jax.tree_util import tree_map, tree_multimap

from boxLCD import env_map
from research import data, runners, utils
from research.define_config import args_type, config, env_fn
from research.nets import net_map

np.cat = np.concatenate
from gym.vector.async_vector_env import AsyncVectorEnv

from research.rl.ppo import _G
from research.rl.pponets import ActorCritic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, value in config().items():
        parser.add_argument(f'--{key}', type=args_type(value), default=value)
    for key, value in _G.items():
        parser.add_argument(f'--{key}', type=args_type(value), default=value)
    temp_cfg = parser.parse_args()
    # grab defaults from the env
    Env = env_map[temp_cfg.env]
    parser.set_defaults(**Env.ENV_DG, **{'num_envs': 10})
    G = parser.parse_args()

    tenv = AsyncVectorEnv([env_fn(G) for _ in range(G.num_envs)])
    env = env_fn(G)()
    if env.__class__.__name__ == 'BodyGoalEnv':
        goal_key = 'goal:proprio'
    elif env.__class__.__name__ == 'CubeGoalEnv':
        goal_key = 'goal:object'
    mG = torch.load(G.weightdir / 'ppo_ac.pt')['G']
    mG.lenv = 0
    ac = ActorCritic(env.observation_space, env.action_space, goal_key, G=mG).to(
        G.device
    )
    ac.load(G.weightdir)
    print('LOADED PPO', G.weightdir)

    logger = defaultdict(lambda: [])

    def test_agent():
        o, ep_ret, ep_len = tenv.reset(), np.zeros(G.num_envs), np.zeros(G.num_envs)
        # run
        ep_done = np.zeros_like(ep_ret)
        success = np.zeros_like(ep_ret)
        for i in range(G.ep_len):
            # Take deterministic actions at test time
            a, v, logp = ac.step(o)
            o, r, d, info = tenv.step(a)
            ep_done = np.logical_or(ep_done, d)
            if i != (G.ep_len - 1):
                success = np.logical_or(
                    success, d
                )  # if they terminate before ep is over, that is a win
            # once the episode is done, stop counting it. we want to make sure there are an equal number of eps always run.
            ep_ret += r * ~ep_done
            ep_len += 1 * ~ep_done
            if np.all(ep_done):
                break
        logger[f'EpRet'] += [ep_ret.mean()]
        logger[f'EpLen'] += [ep_len.mean()]
        logger[f'success_rate'] += [success.mean()]

    N = 100
    for i in range(N):
        test_agent()
    print(f'N = {N*10}')
    print('dict(')
    for key in logger:
        print(f'{key}={np.mean(logger[key])}', end=',')
    print(')')
