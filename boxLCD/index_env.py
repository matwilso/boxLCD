import time
import itertools
from collections import defaultdict
import copy
import sys
import math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, frictionJointDef, contactListener, revoluteJointDef)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import utils

class IndexEnv(gym.Env, EzPickle):
  """
  This class holds the logic for enabling key indexable observation and action.

  Instead of just vectors for obs and act, each index has an associated string name.
  """

  def __init__(self):
    EzPickle.__init__(self)

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def pack_info(self):
    """take self.obs_info and self.act_info and pack them into the gym interface"""
    self.obs_info = utils.sortdict(self.obs_info)
    self.obs_size = len(self.obs_info)
    self.obs_keys = list(self.obs_info.keys())
    self.observation_space = spaces.Box(-1, +1, (self.obs_size,), dtype=np.float32)
    self.act_info = utils.sortdict(self.act_info)
    self.act_size = len(self.act_info)
    self.act_keys = list(self.act_info.keys())
    self.action_space = spaces.Box(-1, +1, (self.act_size,), dtype=np.float32)

  def obs_index(self, name):
    return self.obs_keys.index(name)

  def act_index(self, name):
    return self.act_keys.index(name)

  def get_act_dict(self, act_vec, map=True):
    """convert vector action to dictionary action
    (-1, 1) in vec shape --> (-act_bounds, act_bounds) in dict shape
    """
    act_dict = {}
    for i, key in enumerate(self.act_keys):
      bounds = self.act_info[key]
      act_dict[key] = utils.mapto(act_vec[i], bounds) if map else act_vec[i]
    assert sorted(act_dict) == list(act_dict.keys())
    return act_dict

  def get_act_vec(self, act_dict, map=True):
    """convert act obs to vector act (inverse of get_act_dict)
    (-act_bounds, act_bounds) in dict shape --> (-1, 1) in vector shape
    """
    act_dict = utils.sortdict(act_dict)
    act_vec = []
    for key in act_dict:
      bounds = self.act_info[key]
      val = utils.rmapto(act_dict[key], bounds) if map else act_dict[key]
      act_vec.append(val)
    return np.array(act_vec)

  def get_obs_vec(self, obs_dict, map=True):
    """convert dict obs to vector obs for export
    (-obs_bounds, obs_bounds) in dict shape --> (-1, 1) in vector shape
    """
    obs_dict = utils.sortdict(obs_dict)
    obs_vec = []
    for key in obs_dict:
      bounds = self.obs_info[key]
      val = utils.rmapto(obs_dict[key], bounds) if map else obs_dict[key]
      obs_vec.append(val)
    return np.array(obs_vec)

  def get_obs_dict(self, obs_vec, map=True):
    """convert vector obs to dict obs (reverse of get_obs_vec)
    (-1, 1) in vec shape --> (-obs_bounds, obs_bounds) in dict shape
    """
    obs_dict = {}

    for i, key in enumerate(self.obs_keys):
      bounds = self.obs_info[key]
      try:
        obs_dict[key] = utils.mapto(obs_vec[i], bounds) if map else obs_vec[i]
      except:
        import ipdb; ipdb.set_trace()
    assert sorted(obs_dict) == list(obs_dict.keys())
    return obs_dict

  def map_dict_obs(self, obs_dict):
    """dict: (-1, 1) --> bounds"""
    new_dict = {}
    for key in obs_dict:
      bounds = self.obs_info[key]
      new_dict[key] = utils.mapto(obs_dict[key], bounds)
    return new_dict

  def unmap_dict_obs(self, obs_dict):
    """dict: bounds --> (-1, 1)"""
    new_dict = {}
    for key in obs_dict:
      bounds = self.obs_info[key]
      new_dict[key] = utils.rmapto(obs_dict[key], bounds)
    return new_dict