from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from inspect import isclass
from ._base import VideoModel

video_model_map = {}
package_dir = str(Path(__file__).resolve().parent)
for (_, module_name, _) in iter_modules([package_dir]):
  module = import_module(f'{__name__}.{module_name}')
  for attribute_name in dir(module):
    attribute = getattr(module, attribute_name)
    if isclass(attribute) and issubclass(attribute, VideoModel):
      video_model_map[attribute_name] = attribute
