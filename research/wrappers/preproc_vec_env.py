import copy
from re import I

import gym
import numpy as np
import torch
from gym.utils import EzPickle, seeding
from jax.tree_util import tree_map, tree_multimap
from scipy.spatial.distance import cosine

from research import utils


class PreprocVecEnv:
    """
    Learned model that preprocesses observations and produces a `zstate`
    """

    def __init__(self, model, env, G, device='cuda'):
        self.model = model
        self._env = env
        self.SCALE = 2
        self.G = G
        self.device = device
        self.model.to(device)
        self.model.eval()
        # self.shared_memory = env.shared_memory
        if self.G.learned_rew and 'Cube' in self.G.env:
            if self.G.arbiterdir.name != '':
                arbiter_path = list(self.G.arbiterdir.glob('*.pt'))
                if len(arbiter_path) > 0:
                    arbiter_path = arbiter_path[0]
                self.obj_loc = torch.jit.load(str(arbiter_path))
                self.obj_loc.eval()
                print('LOADED OBJECT LOCALIZER')
            else:
                self.obj_loc = None

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        # import ipdb; ipdb.set_trace()
        base_space = copy.deepcopy(self._env.observation_space)
        base_space.spaces['zstate'] = gym.spaces.Box(-1, 1, (self.model.z_size,))
        if 'goal:full_state' in base_space.spaces:
            base_space.spaces['goal:zstate'] = gym.spaces.Box(
                -1, 1, (self.model.z_size,)
            )
        return base_space

    def _preproc_obs(self, obs):
        batch_obs = {
            key: torch.as_tensor(1.0 * val).float().to(self.device)
            for key, val in obs.items()
        }
        zstate = self.model.encode(batch_obs, noise=False, quantize=False)
        obs['zstate'] = zstate.detach().cpu().numpy()
        if 'goal:full_state' in batch_obs:
            goal = utils.filtdict(batch_obs, 'goal:', fkey=lambda x: x[5:])
            zgoal = self.model.encode(goal, noise=False)
            obs['goal:zstate'] = zgoal.detach().cpu().numpy()
        return obs

    def reset(self, *args, **kwargs):
        obs = self._env.reset(*args, **kwargs)
        if 'RewardLenv' in self._env.__class__.__name__:
            self.last_obs = tree_map(lambda x: np.array(x.cpu().numpy()), obs)
        else:
            self.last_obs = tree_map(lambda x: np.array(x), obs)
        self.last_done = np.zeros(self.G.num_envs)
        return self._preproc_obs(obs)

    def render(self, *args, **kwargs):
        self._env.render(*args, **kwargs)

    def comp_rew(self, z, gz):
        cos = np.zeros(z.shape[0])
        for i in range(len(z)):
            cos[i] = -cosine(z[i], gz[i])
        return cos

    def learned_rew(self, obs, info={}):
        assert 'Cube' in self.G.env, 'ya gotta'
        done = torch.zeros(obs['lcd'].shape[0]).to(self.G.device)
        obs = tree_map(lambda x: torch.as_tensor(1.0 * x).float().to(self.G.device), obs)
        obj = self.obj_loc(obs).detach()
        goal = self.obj_loc(utils.filtdict(obs, 'goal:', fkey=lambda x: x[5:])).detach()
        delta = (obj - goal).abs().mean(-1)
        info['goal_delta'] = (obs['goal:object'] - goal).abs().mean().cpu().detach()
        if self.G.diff_delt:
            last_obs = tree_map(
                lambda x: torch.as_tensor(1.0 * x).float().to(self.G.device), self.last_obs
            )
            last_obj = self.obj_loc(last_obs).detach()
            last_delta = (last_obj - goal).abs().mean(-1)
            rew = -0.05 + 10 * (last_delta - delta)
        else:
            rew = -delta
        done[delta < 0.04] = 1
        rew[delta < 0.04] += 1.0
        return rew.detach().cpu().numpy(), done.detach().cpu().numpy().astype(np.bool)

    def step(self, action):
        obs, rew, done, _info = self._env.step(action)
        obs = self._preproc_obs(obs)
        if self.G.preproc_rew:
            rew = self.comp_rew(obs['zstate'], obs['goal:zstate'])
        elif self.G.learned_rew:
            # self.info = info
            info = {}
            # preproc_rew = self.comp_rew(obs['zstate'], obs['goal:zstate'])
            info['og_rew'] = rew
            # if np.any(done):
            #  import ipdb; ipdb.set_trace()
            rew, _ = self.learned_rew(obs, info)
            # info['preproc_rew'] = preproc_rew
            if 'RewardLenv' in self._env.__class__.__name__:
                info['og_rew'] = info['og_rew'].cpu().detach()
                success = torch.logical_and(done, ~_info['timeout'].bool()).cpu().numpy()

                def fx(x):
                    if isinstance(x, np.ndarray):
                        return x
                    else:
                        return x.cpu().numpy()

                obs = tree_map(fx, obs)
            else:
                timeout = []
                for inf in _info:
                    if 'timeout' in inf and inf['timeout']:
                        timeout += [True]
                    else:
                        timeout += [False]
                timeout = np.array(timeout)
                success = np.logical_and(done, ~timeout)
            rew[success] = 1.0
            info['learned_rew'] = rew
            info['rew_delta'] = info['og_rew'] - rew
        self.last_obs = tree_map(lambda x: np.array(x), obs)
        if 'RewardLenv' in self._env.__class__.__name__:
            rew = torch.as_tensor(rew).to(self.G.device)
            done = torch.as_tensor(done).to(self.G.device)
            obs = tree_map(lambda x: torch.as_tensor(x).to(self.G.device), obs)
        return obs, rew, done, info

    def close(self):
        self._env.close()


if __name__ == '__main__':
    import pathlib
    import time

    import matplotlib.pyplot as plt
    import torch
    import utils
    from body_goal import BodyGoalEnv
    from PIL import Image, ImageDraw, ImageFont

    # from research.nets.bvae import BVAE
    from boxLCD import env_map, envs
    from research.nets import net_map
    from research.wrappers.async_vector_env import AsyncVectorEnv

    G = utils.AttrDict()
    G.env = 'UrchinCube'
    G.state_rew = 1
    G.device = 'cpu'
    G.lcd_h = 16
    G.lcd_w = 32
    G.wh_ratio = 2.0
    G.lr = 1e-3
    # G.lcd_base = 32
    G.rew_scale = 1.0
    G.diff_delt = 1
    G.fps = 10
    G.hidden_size = 128
    G.nfilter = 128
    G.vqK = 128
    G.vqD = 128
    G.goal_thresh = 0.01
    env = envs.UrchinCube(G)
    G.fps = env.G.fps
    # model = BVAE(env, G)
    def env_fn(G, seed=None):
        def _make():
            env = envs.Urchin(G)
            env = BodyGoalEnv(env, G)
            return env

        return _make

    def outproc(img):
        return (
            (255 * img[..., None].repeat(3, -1))
            .astype(np.uint8)
            .repeat(8, 1)
            .repeat(8, 2)
        )

    weightdir = 'logs/april22a/autoencoder/RNLDA/UrchinCube/'
    model_name = 'RNLDA'

    sd = torch.load(weightdir / f'{model_name}.pt')
    mG = sd.pop('G')
    mG.device = G.device
    preproc = net_map[G.model](env, mG)
    preproc.to(G.device)
    preproc.load(G.weightdir)
    for p in preproc.parameters():
        p.requires_grad = False
    preproc.eval()

    start = time.time()
    env = AsyncVectorEnv([env_fn(G) for _ in range(8)])
    env = PreprocVecEnv(model, env, G, device='cpu')
    obs = env.reset(np.arange(8))
    lcds = [obs['lcd']]
    glcds = [obs['goal:lcd']]
    for i in range(200):
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        lcds += [obs['lcd']]
        glcds += [obs['goal:lcd']]
    env.close()
    lcds = torch.as_tensor(np.stack(lcds)).flatten(1, 2).cpu().numpy()
    glcds = torch.as_tensor(np.stack(glcds)).flatten(1, 2).cpu().numpy()
    lcds = (1.0 * lcds - 1.0 * glcds + 1.0) / 2.0
    print('dt', time.time() - start)
    utils.write_gif('realtest.gif', outproc(lcds), fps=G.fps)
