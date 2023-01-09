import random
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


from research.nets.autoencoders.diffusion_v2.gaussian_diffusion import GaussianDiffusion
from .video_interface_net import VideoInterfaceNet
from research.nets.video_models._base import VideoModel
class VideoRIN(VideoModel):
    def __init__(self, env, G):
        super().__init__(env, G)
        self.res = G.dst_resolution
        self.temp_res = 16
        self.net = VideoInterfaceNet(resolution=self.res, G=G)
        self.teacher_net = None
        self.diffusion = GaussianDiffusion(
            mean_type=G.mean_type,
            num_steps=G.timesteps,
            sampler=G.sampler,
            teacher_net=self.teacher_net,
            teacher_mode=self.G.teacher_mode,
            sample_cond_w=G.sample_cond_w,
        )
        self.lcd_proc = lambda x: x[:, :, :self.temp_res]
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

        metrics = self.diffusion.training_losses(
            net=partial(self.net.train_fwd, guide=y, self_cond=random.random() < self.G.self_cond),
            x=lcd,
        )
        metrics = {key: val.mean() for key, val in metrics.items()}
        loss = metrics['loss']
        return loss, metrics

    def sample(self, n, y=None, low_res=None):
        with torch.no_grad():
            y = torch.ones((n, 128), device=self.G.device)
            noise = torch.randn((n, 3, self.temp_res, self.res, self.res), device=self.G.device)
            net = partial(self.net.sample_fwd, guide=y, low_res=low_res)
            zs, xs, eps = self.diffusion.sample(net=net, init_x=noise, cond_w=0.5)
            return {self.G.lcd_key: zs[-1], 'zs': zs, 'xs': xs}

    def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        pass

    def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        n = batch[self.G.lcd_key].shape[0]

        decoded = self.sample(n)
        if arbiter is not None:
            proprio = batch['proprio'][:, ::8]
            gen_batch = {self.G.lcd_key: decoded[self.G.lcd_key], 'proprio': proprio}
            gen_az, gen_act = arbiter(gen_batch)

            data_batch = {
                self.G.lcd_key: self.lcd_proc(batch[self.G.lcd_key]),
                'proprio': proprio,
            }
            data_az, data_act = arbiter(data_batch)
            fvd = utils.compute_fid(
                gen_az.detach().cpu().numpy(), data_az.detach().cpu().numpy()
            )
            metrics['eval/unprompted_fvd'] = fvd

        def grid(name, x):
            x = (x + 1.0) / 2.0
            # assert tuple(x.shape) == (4, 3, 4, self.res, self.res)
            assert tuple(x.shape)[:2] == (4, 3)
            x = rearrange(F.pad(x, (0, 1, 0, 1)), 'b c t h w -> c (b h) (t w)')
            writer.add_image(name, x, epoch)

        def vid25(name, x):
            assert x.shape[0] == 25
            x = (x + 1.0) / 2.0
            assert tuple(x.shape) == (25, 3, self.temp_res, self.res, self.res)
            # add lines between items in the grid and reshape to grid
            vid = rearrange(
                F.pad(x, (0, 1, 0, 1)), '(n1 n2) c t h w -> t c (n1 h) (n2 w)', n1=5, n2=5
            )[None]
            # add a blank frame at the end
            vid = F.pad(vid, (0, 0, 0, 0, 0, 0, 0, 1), value=1.0)
            writer.add_video(name, vid, epoch, fps=min(self.temp_res // 2, 60))

        def gridvid2(name, x):
            x = (x + 1.0) / 2.0
            T = x.shape[0]
            assert tuple(x.shape[1:]) == (4, 3, 4, self.res, self.res)
            vid = rearrange(F.pad(x, (0, 1, 0, 1)), 's b c t h w -> s c (b h) (t w)')[
                None
            ]
            writer.add_video(name, vid, epoch, fps=min(T // 2, 60))

        def error_vid(name, x, y):
            x = (x + 1.0) / 2.0
            y = (y + 1.0) / 2.0
            error = (y - x + 1.0) / 2.0
            stack = torch.cat([x, y, error], -2)
            stack = rearrange(stack, 'b c t h w -> t c h (b w)')[None]
            writer.add_video(name, stack, epoch, fps=min(self.temp_res // 2, 60))

        finalx = decoded[self.G.lcd_key]
        batchx = self.lcd_proc(batch[self.G.lcd_key])

        vid25('finalx_vid25', finalx[:25])
        vid25('batchx_vid25', batchx[:25])

        # i guess each row should be a video. so just the first 4 and make a 4x4
        grid('finalx_still', finalx[:4])
        grid('batchx_still', batchx[:4])

        # if low_res is not None:
        #    grid('low_res', low_res[:4])

        genz = decoded['zs'][:, :4, :, :4]
        genx = decoded['xs'][:, :4, :, :4]
        gridvid2('genz', genz)
        gridvid2('genx', genx)
