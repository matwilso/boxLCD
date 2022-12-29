from functools import partial
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from research.nets.autoencoders.diffusion_v2.gaussian_diffusion import GaussianDiffusion
#from .simple_unet3d import SimpleUnet3D
from .iso_net3d import IsoNet3D as SimpleUnet3D
from research.nets.video_models._base import VideoModel

class VideoDiffusion(VideoModel):
    def __init__(self, env, G):
        super().__init__(env, G)
        self.res = G.dst_resolution
        self.src_res = G.src_resolution
        self.temporal_res = 4

        if self.G.diffusion_mode == 'superres':
            assert (
                G.src_resolution != -1
            ), "you have to set the src resolution for superres model"
            self.superres = True
            self.low_res_key = f'lcd_{self.src_res}'
        else:
            self.superres = False

        self.net = SimpleUnet3D(
            temporal_res=self.temporal_res,
            spatial_res=self.res,
            channels=G.hidden_size,
            dropout=G.dropout,
            superres=self.superres,
        )

        if self.G.teacher_path != Path('.') and self.G.weights_from == Path('.'):
            print("Loading teacher model")
            # initialize student to teacher weights
            self.load_state_dict(torch.load(self.G.teacher_path), strict=False)
            # make teacher itself and freeze it
            self.teacher_net = SimpleUnet3D(G)
            self.teacher_net.load_state_dict(self.net.state_dict().copy())
            self.teacher_net.eval()
            for param in self.teacher_net.parameters():
                param.requires_grad = False
        else:
            self.teacher_net = None

        self.diffusion = GaussianDiffusion(
            mean_type=G.mean_type,
            num_steps=G.timesteps,
            sampler=G.sampler,
            teacher_net=self.teacher_net,
            teacher_mode=self.G.teacher_mode,
            sample_cond_w=G.sample_cond_w,
        )
        self._init()
        #self.proc = lambda x: x[:, :, :4]
        self.proc = lambda x: x[:, :, ::8]
        assert self.G.window == 32
        #self.net.set_attn_masks(iso_image=True)

    @staticmethod
    def upres_coarse(x, low_res, noise=False):
        low_res = F.interpolate(low_res, x.shape[-2:], mode='bilinear')
        if noise:
            low_res = low_res + torch.randn_like(low_res) * 0.01
        return low_res


    def loss(self, batch):
        if np.random.binomial(1, 0.5):
            self.net.set_attn_masks(iso_image=True)
        else:
            self.net.set_attn_masks(iso_image=False)

        lcd = batch[self.G.lcd_key]
        base_lcd = self.proc(lcd)
        y = torch.ones((lcd.shape[0], 128), device=lcd.device)
        low_res = (
            VideoDiffusion.upres_coarse(
                batch[self.G.lcd_key], batch[self.low_res_key], noise=True
            )
            if self.superres
            else None
        )
        metrics = self.diffusion.training_losses(
            net=partial(self.net, guide=y, low_res=low_res), x=base_lcd
        )
        metrics = {key: val.mean() for key, val in metrics.items()}
        loss = metrics['loss']
        return loss, metrics

    def sample(self, n, y=None, low_res=None):
        self.net.eval()
        self.net.set_attn_masks(iso_image=False)
        with torch.no_grad():
            y = torch.ones((n, 128), device=self.G.device)
            noise = torch.randn((n, 3, self.temporal_res, self.res, self.res), device=self.G.device)
            net = partial(self.net, guide=y, low_res=low_res)
            zs, xs, eps = self.diffusion.sample(net=net, init_x=noise, cond_w=0.5)
            return {self.G.lcd_key: zs[-1], 'zs': zs, 'xs': xs}
        self.net.train()

    def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        pass

    def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        n = batch[self.G.lcd_key].shape[0]

        low_res = (
            VideoDiffusion.upres_coarse(
                batch[self.G.lcd_key], batch[self.low_res_key], noise=True
            )
            if self.superres
            else None
        )
        decoded = self.sample(n, low_res=low_res)

        def grid(name, x):
            x = (x + 1.0) / 2.0
            assert tuple(x.shape) == (4, 3, self.temporal_res, self.res, self.res)
            x = rearrange(F.pad(x, (0, 1, 0, 1)), 'b c t h w -> c (b h) (t w)')
            writer.add_image(name, x, epoch)

        def gridvid(name, x):
            x = (x + 1.0) / 2.0
            T = x.shape[0]
            assert tuple(x.shape[1:]) == (4, 3, self.temporal_res, self.res, self.res)
            vid = rearrange(F.pad(x, (0, 1, 0, 1)), 's b c t h w -> s c (b h) (t w)')[None]
            writer.add_video(name, vid, epoch, fps=min(T // 3, 60))

        # i guess each row should be a video. so just the first 4 and make a 4x4
        finalx = decoded[self.G.lcd_key][:4]
        batchx = self.proc(batch[self.G.lcd_key][:4])
        grid('finalx', finalx)
        grid('batchx', batchx)
        if low_res is not None:
            grid('low_res', low_res[:4])
        genz = decoded['zs'][:, :4]
        genx = decoded['xs'][:, :4]
        gridvid('genz', genz)
        gridvid('genx', genx)

