import sys
import inspect
from research.nets import autoencoders, video_models
from .autoencoders._base import Autoencoder
from .video_models._base import VideoModel

net_map = {}
for name, obj in inspect.getmembers(sys.modules[autoencoders.__name__]):
  if inspect.isclass(obj) and issubclass(obj, Autoencoder):
    net_map[name] = obj
for name, obj in inspect.getmembers(sys.modules[video_models.__name__]):
  if inspect.isclass(obj) and issubclass(obj, VideoModel):
    net_map[name] = obj