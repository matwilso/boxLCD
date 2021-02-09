from envs.box import Box
import envs.wrappers as wrappers

object_kwargs = dict(shape='circle', size=2.0, density=0.1)

class Dropbox(B2D):
  def __init__(self, C):
    w = World(agents=[], objects=[Object('object0', shape='box', size=2.0, density=0.1)])
    super().__init__(w, C)

class CrabObject(B2D):
  def __init__(self, C):
    w = World(agents=[Agent(f'{C.cname}0')], objects=[Object('object0', shape='random', size=2.0, density=0.1)])
    super().__init__(w, C)

class Box(B2D):
  def __init__(self, C):
    gravity = [0, -9.81]
    forcetorque = 0
    w = World(agents=[Agent(f'{C.cname}{i}') for i in range(C.num_agents)], objects=[Object(f'object{i}', **object_kwargs) for i in range(C.num_objects)], gravity=gravity, forcetorque=forcetorque)
    super().__init__(w, C)
