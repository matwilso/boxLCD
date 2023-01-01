from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from research import utils
from research.nets.autoencoders.diffusion_v2.gaussian_diffusion import GaussianDiffusion
from research.nets.video_models._base import VideoModel

from .simple_unet3d import SimpleUnet3D

# from .iso_net3d import IsoNet3D as SimpleUnet3D


class VideoDiffusion(VideoModel):
    def __init__(self, env, G):
        super().__init__(env, G)
        self.res = G.dst_resolution
        self.src_res = G.src_resolution
        self.temporal_stride = G.temporal_stride
        self.temporal_res = 32 // self.temporal_stride

        self.superres = False
        self.supertemp = False
        if self.G.diffusion_mode == 'superres':
            assert (
                G.src_resolution != -1
            ), "you have to set the src resolution for superres model"
            self.superres = True
            self.low_res_key = f'lcd_{self.src_res}'
        elif self.G.diffusion_mode == 'supertemp':
            assert (
                G.temporal_stride_src != -1
            ), "you have to set the src temporal stride for supertemp model"
            self.supertemp = True

        self.net = SimpleUnet3D(
            temporal_res=self.temporal_res,
            spatial_res=self.res,
            channels=G.hidden_size,
            dropout=G.dropout,
            superres=self.superres,
            supertemp=self.supertemp,
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
        self.lcd_proc = lambda x: x[:, :, :: self.temporal_stride]
        assert self.G.window == 32
        # self.net.set_attn_masks(iso_image=True)

    @staticmethod
    def upres_coarse(high_res_shape, low_res, noise=False):
        low_res = F.interpolate(low_res, high_res_shape, mode='bilinear')
        if noise:
            low_res = low_res + torch.randn_like(low_res) * 0.01
        return low_res

    @staticmethod
    def tween_coarse(lcd, src_stride, dst_stride, noise=False):
        # this smoothly interpolates between two resolutions
        # basically it creates a onion-skin effect
        base = lcd[:, :, ::src_stride]
        dest_res = 32 // dst_stride
        # if base is shape (B, C, base_res, H, W), we want to get (B, C,dest_res, H, W) by interpolating
        H, W = base.shape[-2:]
        tweened = F.interpolate(base, size=(dest_res, H, W), mode='trilinear')
        if noise:
            tweened = tweened + torch.randn_like(tweened) * 0.01
        return tweened

    def loss(self, batch):
        if np.random.binomial(1, 0.5):
            self.net.set_attn_masks(iso_image=True)
        else:
            self.net.set_attn_masks(iso_image=False)

        lcd = batch[self.G.lcd_key]
        base_lcd = self.lcd_proc(lcd)
        y = torch.ones((lcd.shape[0], 128), device=lcd.device)

        low_res = (
            VideoDiffusion.upres_coarse(
                batch[self.G.lcd_key].shape[-2:], batch[self.low_res_key], noise=True
            )
            if self.superres
            else None
        )

        coarse_tween = (
            VideoDiffusion.tween_coarse(
                batch[self.G.lcd_key],
                self.G.temporal_stride_src,
                self.G.temporal_stride_dst,
                noise=True,
            )
            if self.supertemp
            else None
        )

        metrics = self.diffusion.training_losses(
            net=partial(self.net, guide=y, low_res=low_res, coarse_tween=coarse_tween),
            x=base_lcd,
        )
        metrics = {key: val.mean() for key, val in metrics.items()}
        loss = metrics['loss']
        return loss, metrics

    def sample(self, n, y=None, low_res=None, coarse_tween=None):
        self.net.set_attn_masks(iso_image=False)
        with torch.no_grad():
            y = torch.ones((n, 128), device=self.G.device)
            noise = torch.randn(
                (n, 3, self.temporal_res, self.res, self.res), device=self.G.device
            )
            net = partial(self.net, guide=y, low_res=low_res, coarse_tween=coarse_tween)
            zs, xs, eps = self.diffusion.sample(net=net, init_x=noise, cond_w=0.5)
            out = {self.G.lcd_key: zs[-1], 'zs': zs, 'xs': xs}
        return out

    def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        pass

    def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        n = batch[self.G.lcd_key].shape[0]

        low_res = (
            VideoDiffusion.upres_coarse(
                batch[self.G.lcd_key].shape[-2:], batch[self.low_res_key], noise=True
            )
            if self.superres
            else None
        )

        coarse_tween = (
            VideoDiffusion.tween_coarse(
                batch[self.G.lcd_key],
                self.G.temporal_stride_src,
                self.G.temporal_stride_dst,
                noise=True,
            )
            if self.supertemp
            else None
        )

        decoded = self.sample(n, low_res=low_res, coarse_tween=coarse_tween)
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
            assert tuple(x.shape) == (4, 3, 4, self.res, self.res)
            x = rearrange(F.pad(x, (0, 1, 0, 1)), 'b c t h w -> c (b h) (t w)')
            writer.add_image(name, x, epoch)

        def vid25(name, x):
            assert x.shape[0] == 25
            x = (x + 1.0) / 2.0
            assert tuple(x.shape) == (25, 3, self.temporal_res, self.res, self.res)
            # add lines between items in the grid and reshape to grid
            vid = rearrange(
                F.pad(x, (0, 1, 0, 1)), '(n1 n2) c t h w -> t c (n1 h) (n2 w)', n1=5, n2=5
            )[None]
            # add a blank frame at the end
            vid = F.pad(vid, (0, 0, 0, 0, 0, 0, 0, 1))
            writer.add_video(name, vid, epoch, fps=min(self.temporal_res // 2, 60))

        def gridvid2(name, x):
            x = (x + 1.0) / 2.0
            T = x.shape[0]
            assert tuple(x.shape[1:]) == (4, 3, 4, self.res, self.res)
            vid = rearrange(F.pad(x, (0, 1, 0, 1)), 's b c t h w -> s c (b h) (t w)')[
                None
            ]
            writer.add_video(name, vid, epoch, fps=min(T // 2, 60))

        finalx = decoded[self.G.lcd_key]
        batchx = self.lcd_proc(batch[self.G.lcd_key])

        vid25('finalx_vid25', finalx[:25])
        vid25('batchx_vid25', batchx[:25])
        if coarse_tween is not None:
            vid25('coarse_tween', coarse_tween[:25])

            r = lambda x: rearrange(x, 'b c t h w -> (b t) c h w')
            self.ssim.update((r(finalx), r(batchx)))
            ssim = self.ssim.compute()
            metrics['eval/ssim'] = ssim
            self.psnr.update((r(finalx), r(batchx)))
            psnr = self.psnr.compute().cpu()
            metrics['eval/psnr'] = psnr

        # i guess each row should be a video. so just the first 4 and make a 4x4
        grid('finalx_first4', finalx[:4, :, :4])
        grid('batchx_first4', batchx[:4, :, :4])

        # if low_res is not None:
        #    grid('low_res', low_res[:4])

        genz = decoded['zs'][:, :4, :, :4]
        genx = decoded['xs'][:, :4, :, :4]
        gridvid2('genz', genz)
        gridvid2('genx', genx)
