from utils import A
import utils
from data import data
from envs.box import Box, Dropbox
from define_config import env_fn
from nets.world import AutoWorld

class Runner:
  def __init__(self, C):
    self.env = env_fn(C)()
    self.C = C
    if self.C.model == 'frame_token':
      self.model = AutoWorld(self.env, C)