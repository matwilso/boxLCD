import inspect
import sys
from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

from research.nets import autoencoders, video_models

from ._base import Net
from .autoencoders import autoencoder_map
from .autoencoders._base import Autoencoder
from .video_models import video_model_map
from .video_models._base import VideoModel

net_map = {**autoencoder_map, **video_model_map}
