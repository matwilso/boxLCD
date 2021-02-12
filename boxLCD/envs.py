from boxLCD.world_env import WorldEnv
import boxLCD.wrappers as wrappers
from boxLCD.world_defs import WorldDef, Object, Robot
from boxLCD import utils

# ENVIRONMENT DEFAULT CONFIG
C = utils.AttrDict()
C.base_dim = 5
C.wh_ratio = 1.0
C.lcd_base = 16
C.lcd_render = 1
C.use_images = 0
C.ep_len = 200
C.angular_offset = 0
C.root_offset = 0
C.obj_offset = 0
C.compact_obs = 0
C.use_speed = 1
C.all_contact = 1
C.all_corners = 0
C.walls = 1
C.debug = 0

class Dropbox(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[], objects=[Object('object0', shape='box', size=0.5, density=0.1)])
    C.wh_ratio = 1.0
    C.lcd_base = 16
    C.ep_len = 100
    super().__init__(w, C)

class Bounce(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[], objects=[Object('object0', shape='circle', size=0.7, density=0.1)])
    C.wh_ratio = 1.0
    C.lcd_base = 16
    C.ep_len = 200
    super().__init__(w, C)

class BoxOrCircle(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[], objects=[Object('object0', shape='random', size=0.7, density=0.1, rand_angle=0)])
    C.wh_ratio = 1.0
    C.lcd_base = 16
    C.ep_len = 200
    super().__init__(w, C)

class Urchin(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[])
    C.wh_ratio = 1.0
    C.lcd_base = 16
    super().__init__(w, C)

class UrchinBall(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object('object0', shape='circle', size=0.7, density=0.1)])
    C.wh_ratio = 1.5
    C.lcd_base = 16
    super().__init__(w, C)

class UrchinBalls(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', shape='circle', size=0.7, density=0.1) for i in range(4)])
    C.wh_ratio = 2.0
    C.lcd_base = 16
    super().__init__(w, C)

class UrchinCubes(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', shape='box', size=0.4, density=0.1) for i in range(4)])
    C.wh_ratio = 2.0
    C.lcd_base = 16
    super().__init__(w, C)
