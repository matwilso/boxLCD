import numpy as np
from copy import deepcopy

from gym import logger
from gym.vector.vector_env import VectorEnv
from gym.vector.utils import concatenate, create_empty_array

__all__ = ['SyncVectorEnv']


class SyncVectorEnv(VectorEnv):
  """Vectorized environment that serially runs multiple environments.

  Parameters
  ----------
  env_fns : iterable of callable
      Functions that create the environments.

  observation_space : `gym.spaces.Space` instance, optional
      Observation space of a single environment. If `None`, then the
      observation space of the first environment is taken.

  action_space : `gym.spaces.Space` instance, optional
      Action space of a single environment. If `None`, then the action space
      of the first environment is taken.

  copy : bool (default: `True`)
      If `True`, then the `reset` and `step` methods return a copy of the
      observations.
  """

  def __init__(self, env_fns, observation_space=None, action_space=None, copy=True, C=None):
    self.env_fns = env_fns
    self.envs = [env_fn() for env_fn in env_fns]
    self.copy = copy
    self.rendered = False
    self.C = C

    if (observation_space is None) or (action_space is None):
      observation_space = observation_space or self.envs[0].observation_space
      action_space = action_space or self.envs[0].action_space
    super(SyncVectorEnv, self).__init__(num_envs=len(env_fns),
                                        observation_space=observation_space, action_space=action_space)

    self._check_observation_spaces()
    self.observations = create_empty_array(self.single_observation_space,
                                           n=self.num_envs, fn=np.zeros)
    self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
    self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
    self._actions = None

  def seed(self, seeds=None):
    if seeds is None:
      seeds = [None for _ in range(self.num_envs)]
    if isinstance(seeds, int):
      seeds = [seeds + i for i in range(self.num_envs)]
    assert len(seeds) == self.num_envs

    for env, seed in zip(self.envs, seeds):
      env.seed(seed)

  def reset(self, idxs, phis):
    return self.reset_wait(idxs, phis)

  def render(self, *args, **kwargs):
    imgs = []
    for env in self.envs:
      imgs.append(env.render(*args, **kwargs))
    if not self.rendered:
      self.rendered = True
      if self.C.vanished:
        for env in self.envs:
          if env.viewer is not None:
            env.viewer.window.set_visible(False)
    return np.stack(imgs)

  def reset_wait(self, idxs, phis):
    self._dones[:] = False
    observations = []
    envs = [self.envs[i] for i in idxs]
    for env, phi in zip(envs, phis):
      observation = env.reset(phi)
      observations.append(observation)
    self.observations = concatenate(observations, self.observations, self.single_observation_space)
    return deepcopy(self.observations) if self.copy else self.observations

  def step_async(self, actions):
    self._actions = actions

  def step_wait(self):
    observations, infos = [], []
    for i, (env, action) in enumerate(zip(self.envs, self._actions)):
      observation, self._rewards[i], self._dones[i], info = env.step(action)
      #if self._dones[i]:
      #  observation = env.reset()
      observations.append(observation)
      infos.append(info)
    self.observations = concatenate(observations, self.observations, self.single_observation_space)

    return (deepcopy(self.observations) if self.copy else self.observations, np.copy(self._rewards), np.copy(self._dones), infos)

  def close_extras(self, **kwargs):
    [env.close() for env in self.envs]

  def _check_observation_spaces(self):
    for env in self.envs:
      if not (env.observation_space == self.single_observation_space):
        break
    else:
      return True
    raise RuntimeError('Some environments have an observation space '
                       'different from `{0}`. In order to batch observations, the '
                       'observation spaces from all environments must be '
                       'equal.'.format(self.single_observation_space))
