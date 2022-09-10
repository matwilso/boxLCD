from functools import partial

import torch
import torch.nn.functional as F
import yaml
from einops import parse_shape, rearrange, repeat
from jax.tree_util import tree_map
from torch import nn

from research import utils
from research.nets.autoencoders.video_autoencoder import Decoder
from research.nets.common import (
    ResBlock,
    SelfAttention,
    TimestepEmbedSequential,
    TransformerBlock,
    timestep_embedding,
    zero_module,
)
from research.nets.video_models.gaussian_diffusion import GaussianDiffusion

from ._base import VideoModel

AE_STRIDE = 4  # TODO: replace with the value from loading it
AE_H = 4  # TODO: replace with the value from loading it
AE_W = 4  # TODO: replace with the value from loading it
AE_Z = 32  # TODO: replace with the value from loading it
AE_LOWD = 1  # TODO: replace with the value from loading it


class LatentDiffusionVideo(VideoModel):
    def __init__(self, env, G):
        super().__init__(env, G)
        G.dropout = 0.0
        G.timesteps = 500  # seems to work pretty well for MNIST
        G.timestep_embed = 64
        state_n = env.observation_space.spaces['proprio'].shape[0]
        self.embed_size = G.n_embed
        self.net = SimpleUnet3d(G)
        self.diffusion = GaussianDiffusion(G.timesteps)

        # TODO: maybe consider caching features for a dataset

        # <LOAD ae>
        ae_path = G.weightdir / 'Encoder.pt'
        self.pre_encoder = torch.jit.load(ae_path)
        with (ae_path.parent / 'hps.yaml').open('r') as f:
            aeG = yaml.load(f, Loader=yaml.Loader)
        self.pre_encoder.G = aeG
        for p in self.pre_encoder.parameters():
            p.requires_grad = False
        self.pre_encoder.eval()

        ae_path = G.weightdir / 'Decoder.pt'
        self.pre_decoder = torch.jit.load(ae_path)
        with (ae_path.parent / 'hps.yaml').open('r') as f:
            aeG = yaml.load(f, Loader=yaml.Loader)
        self.pre_decoder.G = aeG
        for p in self.pre_decoder.parameters():
            p.requires_grad = False
        self.pre_decoder.eval()
        # </LOAD ae>
        self._init()

        self.ein_packed = {
            'lcd': '(b d1) c d2 h w',
            'proprio': '(b d1) d2 x',
            'action': '(b d1) d2 x',
            'full_state': '(b d1) d2 x',
        }
        self.ein_unpacked = {
            'lcd': 'b c (d1 d2) h w',
            'proprio': 'b (d1 d2) x',
            'action': 'b (d1 d2) x',
            'full_state': 'b (d1 d2) x',
        }
        self.pack = {
            key: partial(
                rearrange,
                pattern=self.ein_unpacked[key] + '->' + self.ein_packed[key],
                d1=AE_STRIDE,
            )
            for key in self.ein_packed
        }
        self.unpack = {
            key: partial(
                rearrange,
                pattern=self.ein_packed[key] + '->' + self.ein_unpacked[key],
                d1=AE_STRIDE,
            )
            for key in self.ein_packed
        }

    def evaluate(self, epoch, writer, batch, arbiter=None):
        metrics = {}

        if run_sampling := False:
            n = batch['lcd'].shape[0]
            out = self.sample(n, prompts=batch, prompt_n=self.G.prompt_n)
            # self._diffusion_video(epoch, writer, out['diffusion_sampling'], name='diffusion_sampling', prompt_n=None)
            self._diffusion_video(
                epoch,
                writer,
                out['diffusion_pred'],
                name='diffusion_pred',
                prompt_n=None,
            )

        self._prompted_eval(
            epoch, writer, metrics, batch, arbiter, make_video=self.G.make_video
        )
        # self._unprompted_eval(
        #    epoch, writer, metrics, batch, arbiter, make_video=self.G.make_video
        # )
        metrics = tree_map(lambda x: torch.as_tensor(x).cpu(), metrics)
        return metrics

    def _diffusion_video(
        self, epoch, writer, pred, truth=None, name=None, prompt_n=None
    ):
        out = pred[:, :4]  # video_n
        out = torch.cat([out, torch.zeros_like(out[:, :, :, :, :1])], axis=4)
        out = torch.cat([out, torch.zeros_like(out[:, :, :, :, :, :1])], axis=5)
        out = rearrange(out, 's bs c t h w -> s (bs h) (t w) c')
        out = repeat(out, 's h w c -> s (h h2) (w w2) c', h2=2, w2=2)
        utils.add_video(writer, name, out, epoch, fps=60)
        writer.flush()
        print('FLUSH')

    def forward(self, batch):
        from fvcore.nn import (
            ActivationCountAnalysis,
            FlopCountAnalysis,
            activation_count,
            flop_count_str,
            flop_count_table,
            parameter_count,
            parameter_count_table,
        )

        train_batch = self.b(next(train_iter))
        # z = torch.zeros(32, 32, 4, 4, 4).cuda()
        # t = torch.randint(0, self.G.timesteps, (z.shape[0],)).cuda()
        # flops = FlopCountAnalysis(self.model.net, (z, t))
        # flops.total()
        # import ipdb; ipdb.set_trace()

        # TODO: make a flop counter thing
        z = self.preproc(batch)
        t = torch.randint(0, self.G.timesteps, (z.shape[0],)).to(z.device)
        z = self.net(z, t).split(z.shape[1], dim=1)
        out = self.postproc(z)
        return out

    def proc_batch(self, batch):
        return {key: self.pack[key](val) for key, val in batch.items()}

    def preproc(self, batch):
        batch = self.proc_batch(batch)
        z = self.pre_encoder(batch)
        z = self.unpack['lcd'](z)
        return z

    def postproc(self, z):
        assert z.ndim == 5
        dec = self.pre_decoder(self.pack['lcd'](z))
        dec = {key: self.unpack[key](val) for key, val in dec.items()}
        dist = Decoder.dist(dec)
        # TODO: pass through proprio etc as well
        return dist['lcd'].probs

    def loss(self, batch):
        z = self.preproc(batch)
        t = torch.randint(0, self.G.timesteps, (z.shape[0],)).to(z.device)
        metrics = self.diffusion.training_losses(self.net, z, t)
        metrics = {key: val.mean() for key, val in metrics.items()}
        loss = metrics['loss']
        return loss, metrics

    def sample(self, n, action=None, prompts=None, prompt_n=None):
        vid_shape = (n, AE_Z, self.G.window // AE_STRIDE, AE_W, AE_H)
        torch.manual_seed(0)
        noise = torch.randn(vid_shape, device=self.G.device)
        if prompts is not None:
            prompts = self.preproc(prompts)
            prompts = prompts[:, :, : prompt_n // AE_STRIDE]
        all_samples = self.diffusion.p_sample(
            self.net,
            vid_shape,
            noise=noise,
            prompts=prompts,
            prompt_n=prompt_n // AE_STRIDE,
        )
        samps = [al['sample'] for al in all_samples]
        preds = [al['pred_xstart'] for al in all_samples]
        raw_samples = all_samples[-1]['sample']

        raw_samples = self.postproc(raw_samples)

        diffusion_sampling = None
        diffusion_pred = None

        if do_diffusion := True:
            samps = [self.postproc(ds) for ds in samps]
            preds = [self.postproc(dp) for dp in preds]
            diffusion_sampling = torch.stack(samps)
            diffusion_pred = torch.stack(preds)

        return {
            'lcd': raw_samples,
            'diffusion_sampling': diffusion_sampling,
            'diffusion_pred': diffusion_pred,
        }


class SimpleUnet3d(nn.Module):
    def __init__(self, G):
        super().__init__()
        channels = G.hidden_size
        dropout = G.dropout
        time_embed_dim = 2 * channels
        self.time_embed = nn.Sequential(
            nn.Linear(G.timestep_embed, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.down = Down(
            G.window // AE_STRIDE, channels, time_embed_dim, dropout=dropout
        )
        self.turn = ResBlock(channels, time_embed_dim, dropout=dropout)
        self.up = Up(G.window // AE_STRIDE, channels, time_embed_dim, dropout=dropout)
        self.out = nn.Sequential(
            nn.GroupNorm(16, channels),
            nn.SiLU(),
            nn.Conv3d(channels, 2 * AE_Z, (3, 3, 3), padding=(1, 1, 1)),
        )
        self.G = G

    def forward(self, x, timesteps):
        emb = self.time_embed(
            timestep_embedding(
                timesteps.float(), self.G.timestep_embed, timesteps.shape[0]
            )
        )
        # <UNET> downsample, then upsample with skip connections between the down and up.
        x, pass_through = self.down(x, emb)
        x = self.turn(x, emb)
        x = self.up(x, emb, pass_through)
        x = self.out(x)
        # </UNET>
        return x


class Downsample(nn.Module):
    """halve the size of the input"""

    def __init__(self, channels, out_channels=None, stride=2):
        super().__init__()
        out_channels = out_channels or channels
        # TODO: change to no mixing across time
        self.conv = nn.Conv3d(
            channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, stride, stride),
            padding=(1, 1, 1),
        )
        # self.conv = nn.Conv3d(channels, out_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1))

    def forward(self, x, emb=None):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, window, channels, emb_channels, dropout=0.0):
        super().__init__()
        self.seq = nn.ModuleList(
            [
                Downsample(
                    32, channels, stride=1
                ),  # not really a downsample, just makes the code simpler to share
                # ResBlock(channels, emb_channels, dropout=dropout),
                ResBlock(channels, emb_channels, dropout=dropout),
                TimestepEmbedSequential(
                    ResBlock(channels, emb_channels, channels, dropout=dropout),
                    ResBlock(channels, emb_channels, channels, dropout=dropout),
                    AttentionBlock(window, channels),
                ),
                Downsample(channels),
                # ResBlock(channels, emb_channels, dropout=dropout),
                TimestepEmbedSequential(
                    ResBlock(channels, emb_channels, channels, dropout=dropout),
                    ResBlock(channels, emb_channels, channels, dropout=dropout),
                    AttentionBlock(window, channels),
                ),
                ResBlock(channels, emb_channels, dropout=dropout),
                Downsample(channels),
            ]
        )

    def forward(self, x, emb):
        pass_through = []
        for layer in self.seq:
            x = layer(x, emb)
            pass_through += [x]
        return x, pass_through


class Upsample(nn.Module):
    """double the size of the input"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (3, 3, 3), padding=(1, 1, 1))

    def forward(self, x, emb=None):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, window, channels, emb_channels, dropout=0.0):
        super().__init__()
        # on the up, bundle resnets with upsampling so upsampling can be simpler
        self.seq = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
                    Upsample(channels),
                ),
                # ResBlock(2*channels, emb_channels, channels, dropout=dropout),
                ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
                TimestepEmbedSequential(
                    ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
                    AttentionBlock(window, channels),
                ),
                TimestepEmbedSequential(
                    ResBlock(2 * channels, emb_channels, channels), Upsample(channels)
                ),
                ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
                # ResBlock(2*channels, emb_channels, channels, dropout=dropout),
                TimestepEmbedSequential(
                    ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
                    AttentionBlock(window, channels),
                ),
                ResBlock(2 * channels, emb_channels, channels, dropout=dropout),
            ]
        )

    def forward(self, x, emb, pass_through):
        pass_through = pass_through[::-1]
        for i in range(len(self.seq)):
            layer, hoz_skip = self.seq[i], pass_through[i]
            x = torch.cat([x, hoz_skip], 1)
            x = layer(x, emb)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, n, n_embed):
        super().__init__()
        n_head = n_embed // 8
        self.attn = TransformerBlock(n, n_embed, n_head, causal=False)
        # self.attn = SelfAttention(n, n_embed, n_head, causal=False)
        self.register_buffer(
            "time_embed",
            timestep_embedding(torch.linspace(0, 1, n), n_embed, max_period=1).T,
        )

    def forward(self, x, emb=None):
        x = x + self.time_embed[None, :, :, None, None]
        x_shape = parse_shape(x, 'bs c t h w')
        x = rearrange(x, 'bs c t h w -> (bs h w) t c')
        x = self.attn(x)
        x = rearrange(x, '(bs h w) t c -> bs c t h w', **x_shape)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, out_channels=None, dropout=0.0):
        super().__init__()
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(16, channels),
            nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, (3, 3, 3), padding=(1, 1, 1)),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, self.out_channels)
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(16, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv3d(
                    self.out_channels, self.out_channels, (3, 3, 3), padding=(1, 1, 1)
                )
            ),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv3d(
                channels, self.out_channels, 1
            )  # step down size

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)[..., None, None, None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h
