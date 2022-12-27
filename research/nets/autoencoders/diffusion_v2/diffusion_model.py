import random
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from research.nets.autoencoders._base import SingleStepAE

from .gaussian_diffusion import GaussianDiffusion
from .simple_unet import SimpleUnet

# TODO: see about adjusting the noise schedule.


class DiffusionModel(SingleStepAE):
    def __init__(self, env, G):
        super().__init__(env, G)
        self.res = G.dst_resolution
        self.src_res = G.src_resolution
        self.lcd_key = f'lcd_{self.res}'

        if self.G.diffusion_mode == 'superres':
            assert (
                G.src_resolution != -1
            ), "you have to set the src resolution for superres model"
            self.superres = True
            self.low_res_key = f'lcd_{self.src_res}'
        else:
            self.superres = False

        self.net = SimpleUnet(
            resolution=self.res,
            channels=G.hidden_size,
            dropout=G.dropout,
            superres=self.superres,
        )

        if self.G.teacher_path != Path('.') and self.G.weights_from == Path('.'):
            print("Loading teacher model")
            # initialize student to teacher weights
            self.load_state_dict(torch.load(self.G.teacher_path), strict=False)
            # make teacher itself and freeze it
            self.teacher_net = SimpleUnet(G)
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

    def upres_coarse(self, x, low_res):
        return F.interpolate(low_res, x.shape[-2:], mode='bilinear')

    def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        pass

    def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        n = batch[self.lcd_key].shape[0]

        low_res = (
            self.upres_coarse(batch[self.lcd_key], batch[self.low_res_key])
            if self.superres
            else None
        )
        decoded = self.sample(n, low_res=low_res)
        decoded[self.lcd_key] = (decoded[self.lcd_key] + 1.0) / 2.0

        def grid(name, x):
            x = (x + 1.0) / 2.0
            assert tuple(x.shape) == (25, 3, self.res, self.res)
            x = rearrange(x, '(n1 n2) c h w -> c (n1 h) (n2 w)', n1=5, n2=5)
            writer.add_image(name, x, epoch)

        def gridvid(name, x):
            x = (x + 1.0) / 2.0
            T = x.shape[0]
            assert tuple(x.shape[1:]) == (25, 3, self.res, self.res)
            vid = rearrange(x, 't (n1 n2) c h w -> t c (n1 h) (n2 w)', n1=5, n2=5)[None]
            vid = repeat(vid, 'b t c h w -> b t (repeat c) h w', repeat=3)
            writer.add_video(name, vid, epoch, fps=min(T // 3, 60))

        genz = decoded['zs'][:, :25]
        genx = decoded['xs'][:, :25]
        batchx = batch[self.lcd_key][:25]
        gridvid('genz', genz)
        gridvid('genx', genx)
        grid('batchx', batchx)
        grid('finalx', genx[-1, :25])
        if low_res is not None:
            grid('low_res', low_res[:25])

        if arbiter is not None:
            decoded[self.lcd_key] = self.proc(decoded[self.lcd_key])
            paz = arbiter.forward(decoded).cpu().numpy()
            taz = arbiter.forward(batch).cpu().numpy()
            metrics['eval/fid'] = utils.compute_fid(paz, taz)

    def loss(self, batch):
        lcd = batch[self.lcd_key]
        y = torch.ones((lcd.shape[0], 128), device=lcd.device)
        low_res = (
            self.upres_coarse(batch[self.lcd_key], batch[self.low_res_key])
            if self.superres
            else None
        )
        metrics = self.diffusion.training_losses(
            net=partial(self.net, guide=y, low_res=low_res), x=lcd
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
            return {self.lcd_key: zs[-1], 'zs': zs, 'xs': xs}
