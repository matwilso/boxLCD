# ' ' `-' `-`.`-' `-' `-'  '  `-'
import inspect
import sys

from boxLCD import envs
from boxLCD.utils import AttrDict
from boxLCD.world_defs import Object, Robot, WorldDef
from boxLCD.world_env import WorldEnv

from .__version__ import (
    __author__,
    __author_email__,
    __copyright__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
)

ENV_DG = AttrDict(WorldEnv.ENV_DG)
env_map = {}
for name, obj in inspect.getmembers(sys.modules[envs.__name__]):
    if inspect.isclass(obj) and issubclass(obj, WorldEnv):
        env_map[name] = obj
