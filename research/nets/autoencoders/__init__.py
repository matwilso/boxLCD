from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

from ._base import Autoencoder

autoencoder_map = {}
package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([package_dir.__str__()]):
    module = import_module(f'{__name__}.{module_name}')
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isclass(attribute) and issubclass(attribute, Autoencoder):
            autoencoder_map[attribute_name] = attribute

from .diffusion_v2.diffusion_model import DiffusionModel
autoencoder_map['diffusion_model'] = DiffusionModel