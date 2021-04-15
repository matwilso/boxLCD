from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from inspect import isclass
import sys
import inspect
from research.nets import autoencoders, video_models
from .autoencoders._base import Autoencoder
from .video_models._base import VideoModel
from .autoencoders.bvae import Autoencoder

from .autoencoders import autoencoder_map
from .video_models import video_model_map

net_map = {**autoencoder_map, **video_model_map}

# iterate through the modules in the current package
#net_map = {}
#for name, obj in inspect.getmembers(sys.modules[autoencoders.__name__]):
#  print(name, obj)
#  if inspect.isclass(obj) and issubclass(obj, Autoencoder):
#    net_map[name] = obj
#for name, obj in inspect.getmembers(sys.modules[video_models.__name__]):
#  print(name, obj)
#  if inspect.isclass(obj) and issubclass(obj, VideoModel):
#    net_map[name] = obj