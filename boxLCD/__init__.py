from boxLCD.world_env import WorldEnv, C
import boxLCD.wrappers as wrappers
from boxLCD.world_defs import WorldDef, Object, Robot

class Dropbox(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[], objects=[Object('object0', shape='box', size=0.5, density=0.1)])
    super().__init__(w, C)

class CrabObject(WorldEnv):
  def __init__(self, C):
    w = WorldDef(robots=[Robot(f'{C.cname}0')], objects=[Object('object0', shape='random', size=2.0, density=0.1)])
    super().__init__(w, C)

class Box(WorldEnv):
  def __init__(self, C):
    gravity = [0, -9.81]
    forcetorque = 0
    #object_kwargs = dict(shape='box', size=0.3, density=0.8, friction=1.0)
    object_kwargs = dict(shape='circle', size=0.7, density=0.01)
    num_robots = 1
    num_objects = 1
    w = WorldDef(robots=[Robot('urchin', f'urchin{i}') for i in range(num_robots)], objects=[Object(f'object{i}', **object_kwargs) for i in range(num_objects)], gravity=gravity, forcetorque=forcetorque)
    #w = WorldDef(robots=[Robot('crab', f'crab{i}') for i in range(num_robots)], objects=[Object(f'object{i}', **object_kwargs) for i in range(num_objects)], gravity=gravity, forcetorque=forcetorque)
    super().__init__(w, C)