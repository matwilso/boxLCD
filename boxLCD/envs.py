from boxLCD.world_env import WorldEnv
from boxLCD.world_defs import WorldDef, Object, Robot
from boxLCD import utils

def cc(**kwargs):
  """custom config (default settings)"""
  def decorator(Cls):
    class CustomWorldEnv(Cls):
      # grab the WorldEnv ENV_DC and update it with new defaults
      ENV_DC = utils.AttrDict(WorldEnv.ENV_DC)
      for key in kwargs:
        ENV_DC[key] = kwargs[key]
    return CustomWorldEnv
  return decorator

# BASIC TIER
@cc(ep_len=50, wh_ratio=1.0)
class Dropbox(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[], objects=[Object('object0', shape='box', size=0.7, density=0.1)])
    super().__init__(w, C)

@cc(ep_len=100, wh_ratio=1.0)
class Bounce(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[], objects=[Object('object0', shape='circle', size=0.7, density=0.1)])
    super().__init__(w, C)

@cc(ep_len=100, wh_ratio=1.0)
class BoxOrCircle(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[], objects=[Object('object0', shape='random', size=0.7, density=0.1, rand_angle=0)])
    super().__init__(w, C)

@cc(wh_ratio=1.0)
class Bounce2(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[], objects=[Object(f'object{i}', shape='circle', size=0.7, density=0.1, restitution=0.9) for i in range(2)])
    super().__init__(w, C)

@cc(wh_ratio=1.0)
class Bounce3(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[], objects=[Object(f'object{i}', shape='circle', size=0.7, density=0.1, restitution=0.9) for i in range(3)])
    super().__init__(w, C)


# SIMPLE ROBOTS

class Urchin(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[])
    super().__init__(w, C)

class Luxo(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[])
    super().__init__(w, C)

# SIMPLE ROBOT OBJECT MANIPULATION
#srom_obj_settings = dict(shape='box', size=0.4, density=0.25, linearDamping=1.0, angularDamping=0.2)
srom_obj_settings = dict(shape='box', size=0.4, density=0.25, linearDamping=1.0, angularDamping=0.2)
#srom_obj_settings = dict(shape='box', size=0.4, density=0.1, linearDamping=5.0, angularDamping=1.0)

class UrchinCube(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', **srom_obj_settings) for i in range(1)])
    super().__init__(w, C)

class LuxoCube(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[Object(f'object{i}', **srom_obj_settings) for i in range(1)])
    super().__init__(w, C)

# just for learning model. harder to define a task with these
class UrchinBall(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object('object0', shape='circle', size=0.7, density=0.1)])
    super().__init__(w, C)

class UrchinBalls(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', shape='circle', size=0.7, density=0.1) for i in range(3)])
    super().__init__(w, C)

class UrchinCubes(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', shape='box', size=0.4, density=0.1) for i in range(3)])
    super().__init__(w, C)

class LuxoBall(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[Object('object0', shape='circle', size=0.7, density=0.1)])
    super().__init__(w, C)

# MORE ADVANCED

class Crab(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='crab', name='crab0')])
    super().__init__(w, C)

class CrabCube(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='crab', name='crab0')], objects=[Object(f'object{i}', shape='box', size=0.4, density=1.0, friction=1.0) for i in range(1)])
    super().__init__(w, C)

#class QuadCube(WorldEnv):
#  def __init__(self, C={}):
#    w = WorldDef(robots=[Robot(type='quad', name='quad0')], objects=[Object(f'object{i}', shape='box', size=0.3, density=0.1, friction=1.0) for i in range(1)])
#    super().__init__(w, C)

class SpiderCube(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='spider', name='spider0')], objects=[Object(f'object{i}', shape='box', size=0.3, density=0.1, friction=1.0) for i in range(1)])
    super().__init__(w, C)
