import random
from functools import partial
from pathlib import Path
from einops import reduce, rearrange, repeat

import torch

from .gaussian_diffusion import GaussianDiffusion
from .simple_unet import SimpleUnet
from research.nets.autoencoders._base import SingleStepAE

# TODO: see about adjusting the noise schedule.

class DiffusionModel(SingleStepAE):

    def __init__(self, env, G):
        super().__init__(env, G)
        self.net = SimpleUnet(G)
        # if base model, use the resolution provided
        # if a super res model, going to be more complicated
        self.lcd_key = 'lcd_16'
        self.lcd_base = 16

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

    def _prompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        pass

    def _unprompted_eval(self, epoch, writer, metrics, batch, arbiter=None):
        n = batch[self.lcd_key].shape[0]
        decoded = self.sample(n)

        decoded[self.lcd_key] = (decoded[self.lcd_key] + 1.0) / 2.0

        def grid(name, x):
            x = (x + 1.0) / 2.0
            assert tuple(x.shape) == (25, 3, self.lcd_base, self.lcd_base)
            x = rearrange(x, '(n1 n2) c h w -> c (n1 h) (n2 w)', n1=5, n2=5)
            writer.add_image(name, x, epoch)

        def gridvid(name, x):
            x = (x + 1.0) / 2.0
            T = x.shape[0]
            assert tuple(x.shape[1:]) == (25, 3, self.lcd_base, self.lcd_base)
            vid = rearrange(x, 't (n1 n2) c h w -> t c (n1 h) (n2 w)', n1=5, n2=5)[None]
            vid = repeat(vid, 'b t c h w -> b t (repeat c) h w', repeat=3)
            writer.add_video(
                name,
                vid,
                epoch,
                fps=min(T // 3, 60)
            )

        genz = decoded['zs'][:, :25]
        genx = decoded['xs'][:, :25]
        batchx = batch[self.lcd_key][:25]
        gridvid('genz', genz)
        gridvid('genx', genx)
        grid('batchx', batchx)

        if self.lcd_key in decoded:
            sample_lcd = decoded[self.lcd_key]
            self._plot_lcds(epoch, writer, sample_lcd)

        if 'proprio' in decoded:
            sample_proprio = decoded['proprio']
            self._plot_proprios(epoch, writer, sample_proprio)

        if arbiter is not None:
            decoded[self.lcd_key] = self.proc(decoded[self.lcd_key])
            paz = arbiter.forward(decoded).cpu().numpy()
            taz = arbiter.forward(batch).cpu().numpy()
            metrics['eval/fid'] = utils.compute_fid(paz, taz)


    def loss(self, batch):
        lcd = batch[self.lcd_key]
        y = torch.ones((lcd.shape[0], 128), device=lcd.device)
        metrics = self.diffusion.training_losses(net=partial(self.net, guide=y), x=lcd)
        metrics = {key: val.mean() for key, val in metrics.items()}
        loss = metrics['loss']
        return loss, metrics

    def sample(self, n, y=None):
        with torch.no_grad():
            y = torch.ones((n, 128), device=self.G.device)
            noise = torch.randn((n, 3, self.lcd_base, self.lcd_base), device=self.G.device)
            net = partial(self.net, guide=y)
            zs, xs, eps = self.diffusion.sample(net=net, init_x=noise, cond_w=0.5)
            return {self.lcd_key: zs[-1], 'zs': zs, 'xs': xs}

    #def evaluate(self, writer, x, y, epoch):
    #    # draw samples and visualize the sampling process
    #    def proc(x):
    #        x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
    #        if self.G.pad32:
    #            x = x[..., 2:-2, 2:-2]
    #        return x

    #    # TODO: show unconditional samples as well
    #    torch.manual_seed(0)
    #    noise = torch.randn((25, 1, self.size, self.size), device=x.device)
    #    labels = torch.arange(25, dtype=torch.long, device=x.device) % 10
    #    zs, xs, eps = self.diffusion.sample(
    #        net=partial(self.net, guide=labels), init_x=noise
    #    )
    #    zs, xs, eps = proc(zs), proc(xs), proc(eps)
    #    sample = zs[-1]
    #    common.write_grid(writer, 'samples', sample, epoch)
    #    common.write_gridvid(writer, 'sampling_process', zs, epoch)
    #    common.write_gridvid(writer, 'diffusion_model/eps', eps, epoch)
    #    common.write_gridvid(writer, 'diffusion_model/x', xs, epoch)
    #    torch.manual_seed(random.randint(0, 2**32))
