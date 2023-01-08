import random
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from research.nets.autoencoders._base import SingleStepAE

from .gaussian_diffusion import GaussianDiffusion
from .interface_net import InterfaceNet


class RIN(SingleStepAE):
    def __init__(self, env, G):
        super().__init__(env, G)
        hidden_size = G.hidden_size
        self.net = InterfaceNet(resolution=G.dst_resolution, hidden_size=hidden_size)
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

    def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        pass

    def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        n = batch[self.G.lcd_key].shape[0]
        decoded = self.sample(n)
        decoded[self.G.lcd_key] = (decoded[self.G.lcd_key] + 1.0) / 2.0

        def grid(name, x):
            x = (x + 1.0) / 2.0
            assert tuple(x.shape) == (25, 3, self.res, self.res)
            x = rearrange(
                F.pad(x, (0, 1, 0, 1)), '(n1 n2) c h w -> c (n1 h) (n2 w)', n1=5, n2=5
            )
            writer.add_image(name, x, epoch)

        def gridvid(name, x):
            x = (x + 1.0) / 2.0
            T = x.shape[0]
            assert tuple(x.shape[1:]) == (25, 3, self.res, self.res)
            vid = rearrange(
                F.pad(x, (0, 1, 0, 1)), 't (n1 n2) c h w -> t c (n1 h) (n2 w)', n1=5, n2=5
            )[None]
            vid = repeat(vid, 'b t c h w -> b t (repeat c) h w', repeat=3)
            writer.add_video(name, vid, epoch, fps=min(T // 3, 60))

        genz = decoded['zs'][:, :25]
        genx = decoded['xs'][:, :25]
        batchx = batch[self.G.lcd_key][:25]
        gridvid('genz', genz)
        gridvid('genx', genx)
        grid('batchx', batchx)
        grid('finalx', genx[-1, :25])

    def loss(self, batch):
        lcd = batch[self.G.lcd_key]
        y = torch.ones((lcd.shape[0], 128), device=lcd.device)

        # ok what's the best way to do this
        # i feel like we could create a function wrapper on our net, that takes whatever inputs it would have
        # taken and sometimes runs twice. so yeah...
        # and then it doesn't return z. and then we have a different wrapper for inference

        metrics = self.diffusion.training_losses(
            net=partial(self.net.train_fwd, guide=y, self_cond=random.random() < 0.5),
            x=lcd,
        )
        metrics = {key: val.mean() for key, val in metrics.items()}
        loss = metrics['loss']
        return loss, metrics

    def sample(self, n, y=None, low_res=None):
        with torch.no_grad():
            y = torch.ones((n, 128), device=self.G.device)
            noise = torch.randn((n, 3, self.res, self.res), device=self.G.device)
            net = partial(self.net, guide=y, low_res=low_res)
            zs, xs, eps = self.diffusion.sample(net=net, init_x=noise, cond_w=0.5)
            return {self.G.lcd_key: zs[-1], 'zs': zs, 'xs': xs}
