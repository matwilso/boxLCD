from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from inspect import isclass
import sys
import inspect
from research.nets import autoencoders, video_models
from .autoencoders._base import Autoencoder
from .video_models._base import VideoModel

from .autoencoders import autoencoder_map
from .video_models import video_model_map

from ._base import Net
net_map = {**autoencoder_map, **video_model_map}