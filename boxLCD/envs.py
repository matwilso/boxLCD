from boxLCD.world_env import WorldEnv
import boxLCD.wrappers as wrappers
from boxLCD.world_defs import WorldDef, Object, Robot
from boxLCD import utils

# ENVIRONMENT DEFAULT CONFIG
C = utils.AttrDict()
C.base_dim = 5  # base size of box2D physics world
C.wh_ratio = 1.0  # width:height ratio of the world and images
C.lcd_base = 16  # base size of lcd rendered image. this represents the height. width = wh_ratio*height
C.ep_len = 200  # length to run episode before done timeout
# settings for different obs and action spaces
C.angular_offset = 0  # compute joint angular offsets from robot roots
C.root_offset = 0  # compute position offsets from root
C.compact_obs = 0  # use compact joint angle space instead of joint positions and sin+cos of theta
C.use_speed = 1  # use velocity control vs. torque control
C.all_corners = 0  # use corner keypoint obs instead of sin+cos of theta
C.walls = 1  # bound the environment with walls on both sides
C.debug = 0

class Dropbox(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[], objects=[Object('object0', shape='box', size=0.5, density=0.1)])
    C.wh_ratio = 1.0
    C.ep_len = 100
    super().__init__(w, C)

class Bounce(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[], objects=[Object('object0', shape='circle', size=0.7, density=0.1)])
    C.wh_ratio = 1.0
    C.ep_len = 200
    super().__init__(w, C)

class BoxOrCircle(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[], objects=[Object('object0', shape='random', size=0.7, density=0.1, rand_angle=0)])
    C.wh_ratio = 1.0
    C.ep_len = 200
    super().__init__(w, C)

class Urchin(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[])
    C.wh_ratio = 1.0
    super().__init__(w, C)

class UrchinBall(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object('object0', shape='circle', size=0.7, density=0.1)])
    C.wh_ratio = 1.5
    super().__init__(w, C)

class UrchinBalls(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', shape='circle', size=0.7, density=0.1) for i in range(4)])
    C.wh_ratio = 2.0
    super().__init__(w, C)

class UrchinCubes(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', shape='box', size=0.4, density=0.1) for i in range(4)])
    C.wh_ratio = 2.0
    super().__init__(w, C)