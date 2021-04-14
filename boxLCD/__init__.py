import sys
import inspect
from boxLCD.world_env import WorldEnv
from boxLCD.world_defs import WorldDef, Object, Robot
from boxLCD import envs
from boxLCD.utils import AttrDict
ENV_DG = AttrDict(WorldEnv.ENV_DG)
env_map = {}
for name, obj in inspect.getmembers(sys.modules[envs.__name__]):
  if inspect.isclass(obj) and issubclass(obj, WorldEnv):
    env_map[name] = obj