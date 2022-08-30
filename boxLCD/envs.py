from boxLCD.world_env import WorldEnv
from boxLCD.world_defs import WorldDef, Object, Robot
from boxLCD import utils

def cc(**kwargs):
  """custom config (default settings)"""
  def decorator(Cls):
    class CustomWorldEnv(Cls):
      # grab the WorldEnv ENV_DG and update it with new defaults
      ENV_DG = utils.AttrDict(WorldEnv.ENV_DG)
      for key in kwargs:
        ENV_DG[key] = kwargs[key]
    return CustomWorldEnv
  return decorator

# BASIC PASSIVE ENVS
@cc(ep_len=25, wh_ratio=1.0)
class Dropbox(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[], objects=[Object('object0', shape='box', size=0.7, density=0.1)])
    super().__init__(w, G)

@cc(ep_len=50, wh_ratio=1.0)
class Bounce(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[], objects=[Object('object0', shape='circle', size=0.5, density=0.1, restitution=0.8)])
    super().__init__(w, G)

@cc(ep_len=50, wh_ratio=1.0)
class BounceColor(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[], objects=[Object('object0', shape='circle', size=0.5, density=0.1, restitution=0.8, color='random')], background_color='random', render_mode='rgb')
    super().__init__(w, G)

@cc(ep_len=50, wh_ratio=1.0)
class Bounce2(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[], objects=[Object(f'object{i}', shape='circle', size=0.5, density=0.1, restitution=0.8) for i in range(2)])
    super().__init__(w, G)

@cc(ep_len=50, wh_ratio=1.0)
class Object2(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[], objects=[Object(f'object{i}', shape='random', size=0.5, density=0.1, restitution=0.8) for i in range(2)])
    super().__init__(w, G)

@cc(ep_len=50, wh_ratio=1.0)
class Object3(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[], objects=[Object(f'object{i}', shape='random', size=0.5, density=0.1, restitution=0.8) for i in range(3)])
    super().__init__(w, G)

# SIMPLE ROBOTS
@cc(ep_len=100)
class Urchin(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[])
    super().__init__(w, G)

@cc(ep_len=100)
class Luxo(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[])
    super().__init__(w, G)

# SIMPLE ROBOT OBJECT MANIPULATION
#srom_obj_settings = dict(shape='box', size=0.4, density=0.25, linearDamping=1.0, angularDamping=0.2)
#srom_obj_settings = dict(shape='box', size=0.4, density=0.1, linearDamping=5.0, angularDamping=1.0)
cube_settings = dict(shape='box', size=0.4, density=0.5, linearDamping=1.0, angularDamping=0.2)
ball_settings = dict(shape='circle', size=0.5, density=0.2, restitution=0.8)

@cc(ep_len=150, wh_ratio=1.5)
class UrchinCube(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', **cube_settings) for i in range(1)])
    super().__init__(w, G)

@cc(ep_len=150, wh_ratio=1.5)
class LuxoCube(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[Object(f'object{i}', **cube_settings) for i in range(1)])
    super().__init__(w, G)

# just for learning model. harder to define a task with these
@cc(ep_len=150, wh_ratio=1.5)
class UrchinBall(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object('object0', **ball_settings)])
    #w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object('object0', shape='circle', size=0.4, density=0.2, restitution=0.8)])
    super().__init__(w, G)

@cc(ep_len=150, wh_ratio=1.5)
class LuxoBall(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[Object('object0', **ball_settings)])
    super().__init__(w, G)

class UrchinBalls(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', **ball_settings) for i in range(3)])
    super().__init__(w, G)

class LuxoBalls(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[Object(f'object{i}', **ball_settings) for i in range(3)])
    super().__init__(w, G)

class UrchinCubes(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', **cube_settings) for i in range(3)])
    super().__init__(w, G)

class LuxoCubes(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[Object(f'object{i}', **cube_settings) for i in range(3)])
    super().__init__(w, G)

@cc(ep_len=150, wh_ratio=1.5)
class UrchinShapes(WorldEnv):
  def __init__(self, G={}):
    objects = [Object(f'object{i}', shape='random', size=0.5, density=0.1, restitution=0.8, dropout=0.5, color='random') for i in range(4)]
    robots = [Robot(type='urchin', name='urchin0', color='random')]
    w = WorldDef(robots=robots, objects=objects, background_color='random')
    super().__init__(w, G)

# MORE ADVANCED
# (still being designed. and you might want to run these at higher FPS/Hz)

@cc(lcd_base=32)
class Crab(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='crab', name='crab0')])
    super().__init__(w, G)

@cc(lcd_base=32)
class CrabCube(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='crab', name='crab0')], objects=[Object(f'object{i}', shape='box', size=0.4, density=1.0, friction=1.0) for i in range(1)])
    super().__init__(w, G)

#class QuadCube(WorldEnv):
#  def __init__(self, G={}):
#    w = WorldDef(robots=[Robot(type='quad', name='quad0')], objects=[Object(f'object{i}', shape='box', size=0.3, density=0.1, friction=1.0) for i in range(1)])
#    super().__init__(w, G)

@cc(lcd_base=32)
class SpiderCube(WorldEnv):
  def __init__(self, G={}):
    w = WorldDef(robots=[Robot(type='spider', name='spider0')], objects=[Object(f'object{i}', shape='box', size=0.3, density=0.1, friction=1.0) for i in range(1)])
    super().__init__(w, G)
