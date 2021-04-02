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

@cc(ep_len=50)
class Dropbox(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[], objects=[Object('object0', shape='box', size=0.7, density=0.1)])
    super().__init__(w, C)

@cc(ep_len=100)
class Bounce(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[], objects=[Object('object0', shape='circle', size=0.7, density=0.1)])
    super().__init__(w, C)

@cc(ep_len=100)
class BoxOrCircle(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[], objects=[Object('object0', shape='random', size=0.7, density=0.1, rand_angle=0)])
    super().__init__(w, C)

class Urchin(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[])
    super().__init__(w, C)

@cc(wh_ratio=1.5)
class UrchinBall(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object('object0', shape='circle', size=0.7, density=0.1)])
    super().__init__(w, C)

@cc(wh_ratio=2.0)
class UrchinBalls(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', shape='circle', size=0.7, density=0.1) for i in range(3)])
    super().__init__(w, C)

@cc(wh_ratio=2.0)
class UrchinCubes(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', shape='box', size=0.4, density=0.1) for i in range(3)])
    super().__init__(w, C)

class Bounce2(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[], objects=[Object(f'object{i}', shape='circle', size=0.7, density=0.1, restitution=0.9) for i in range(2)])
    super().__init__(w, C)

class Legs(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='legs', name='legs0')], objects=[])
    super().__init__(w, C)

@cc(wh_ratio=1.5)
class Luxo(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[])
    super().__init__(w, C)

@cc(wh_ratio=1.5)
class LuxoBall(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[Object('object0', shape='circle', size=0.7, density=0.1)])
    super().__init__(w, C)


@cc(wh_ratio=2.0)
class UrchinCube(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='urchin', name='urchin0')], objects=[Object(f'object{i}', shape='box', size=0.4, density=0.1, friction=1.0) for i in range(1)])
    super().__init__(w, C)

@cc(wh_ratio=2.0)
class LuxoCube(WorldEnv):
  def __init__(self, C={}):
    w = WorldDef(robots=[Robot(type='luxo', name='luxo0')], objects=[Object(f'object{i}', shape='box', size=0.4, density=0.1) for i in range(1)])
    super().__init__(w, C)

