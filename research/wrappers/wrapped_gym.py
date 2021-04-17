import gym
import numpy as np

class WrappedGym:
  def __init__(self, env, G):
    self._env = env

  @property
  def action_space(self):
    return self._env.action_space

  @property
  def observation_space(self):
    spaces = {}
    spaces['proprio'] = spaces['full_state'] = self._env.observation_space 
    spaces['goal:proprio'] = spaces['goal:full_state'] = gym.spaces.Box(-1, 1, (1,))
    return gym.spaces.Dict(spaces)

  def reset(self, *args, **kwargs):
    self.goal = {'goal:proprio': np.zeros(1), 'goal:full_state': np.zeros(1)}
    obs = self._env.reset()
    return {'proprio': obs, 'full_state': obs, **self.goal}

  def render(self, *args, **kwargs):
    self._env.render(*args, **kwargs)

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    obs = {'proprio': obs, 'full_state': obs, **self.goal}
    return obs, rew, done, info

  def close(self):
    self._env.close()

  def seed(self, seed):
    self._env.seed(seed)