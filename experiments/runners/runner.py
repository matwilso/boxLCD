from boxLCD.utils import A
import utils
import data
from define_config import env_fn
from nets.world import AutoWorld

class Runner:
  def __init__(self, C):
    self.env = env_fn(C)()
    self.C = C
    self.lcd_w = int(self.C.lcd_base*self.C.wh_ratio)
    self.lcd_h = self.C.lcd_base
    if self.C.model == 'frame_token':
      self.model = AutoWorld(self.env, C)