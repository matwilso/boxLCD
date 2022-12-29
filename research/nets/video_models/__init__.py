from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

from ._base import VideoModel
from .video_diffusion.video_diffusion import VideoDiffusion

video_model_map = {}
package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([package_dir.__str__()]):
    module = import_module(f'{__name__}.{module_name}')
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isclass(attribute) and issubclass(attribute, VideoModel):
            video_model_map[attribute_name] = attribute

# from diffusion.latent_diffusion_video import LatentDiffusionVideo
# from research.nets.video_models.latent_diffusion_video_v2 import LatentDiffusionVideo_v2
#
# import ipdb; ipdb.set_trace()

video_model_map['video_diffusion'] = VideoDiffusion