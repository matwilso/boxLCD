from re import I
from research.nets.video_models.gaussian_diffusion import GaussianDiffusion
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from research.nets.common import TimestepEmbedSequential, ResBlock, timestep_embedding, zero_module
from ._base import VideoModel
from einops import rearrange
from jax.tree_util import tree_map

class DiffusionVideo(VideoModel):
  def __init__(self, env, G):
    super().__init__(env, G)
    G.dropout = 0.0
    G.timesteps = 500 # seems to work pretty well for MNIST
    state_n = env.observation_space.spaces['proprio'].shape[0]
    self.embed_size = 256
    self.net = SimpleUnet3d(G)
    self.diffusion = GaussianDiffusion(G.timesteps)
    self._init()

  def evaluate(self, epoch, writer, batch, arbiter=None):
    metrics = {}
    self._prompted_eval(epoch, writer, metrics, batch, arbiter)
    metrics = tree_map(lambda x: th.as_tensor(x).cpu(), metrics)
    return metrics

  def loss(self, batch):
    lcd = (batch['lcd'][:,None] * 2) - 1
    #lcd[:, :, :] = lcd[:,:,0,None]
    t = th.randint(0, self.G.timesteps, (lcd.shape[0],)).to(lcd.device)
    metrics = self.diffusion.training_losses(self.net, lcd, t)
    metrics = {key: val.mean() for key, val in metrics.items()}
    loss = metrics['loss']
    return loss, metrics

  def sample(self, n, action=None, prompts=None, prompt_n=None):
    vid_shape = (n, 1, self.G.window, self.G.lcd_h, self.G.lcd_w)
    th.manual_seed(0)
    noise = th.randn(vid_shape, device=self.G.device)
    if prompts is not None:
      prompts = (prompts['lcd'][:, None] * 2) - 1
      #prompts[:, :, :] = prompts[0,None,:,0,None]
    all_samples = self.diffusion.p_sample(self.net, vid_shape, noise=noise, prompts=prompts, prompt_n=prompt_n)
    raw_samples = all_samples[-1]['sample']
    raw_samples = (raw_samples + 1) / 2
    return {'lcd': rearrange(raw_samples, 'bs c t h w -> bs t c h w')}


class SimpleUnet3d(nn.Module):
  def __init__(self, G):
    super().__init__()
    channels = G.hidden_size
    dropout = G.dropout
    time_embed_dim = 2 * channels
    self.time_embed = nn.Sequential(
        nn.Linear(64, time_embed_dim),
        nn.SiLU(),
        nn.Linear(time_embed_dim, time_embed_dim),
    )
    self.down = Down(channels, time_embed_dim, dropout=dropout)
    self.turn = ResBlock(channels, time_embed_dim, dropout=dropout)
    self.up = Up(channels, time_embed_dim, dropout=dropout)
    self.out = nn.Sequential(
      nn.GroupNorm(8, channels),
      nn.SiLU(),
      nn.Conv3d(channels, 2, (3, 3, 3), padding=(1, 1, 1)),
    )
    self.G = G

  def forward(self, x, timesteps):
    emb = self.time_embed(timestep_embedding(timesteps.float(), 64, timesteps.shape[0]))
    # <UNET> downsample, then upsample with skip connections between the down and up.
    x, cache = self.down(x, emb)
    x = self.turn(x, emb)
    x = self.up(x, emb, cache)
    x = self.out(x)
    # </UNET>
    return x

class Downsample(nn.Module):
  """halve the size of the input"""
  def __init__(self, channels, out_channels=None, stride=2):
    super().__init__()
    out_channels = out_channels or channels
    # TODO: change to no mixing across time
    self.conv = nn.Conv3d(channels, out_channels, kernel_size=(3, 3, 3), stride=(1, stride, stride), padding=(1, 1, 1))
    #self.conv = nn.Conv3d(channels, out_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1))

  def forward(self, x, emb=None):
    return self.conv(x)

class Down(nn.Module):
  def __init__(self, channels, emb_channels, dropout=0.0):
    super().__init__()
    self.seq = nn.ModuleList([
        Downsample(1, channels, 1),  # not really a downsample, just makes the code simpler to share
        ResBlock(channels, emb_channels, dropout=dropout),
        ResBlock(channels, emb_channels, dropout=dropout),
        Downsample(channels),
        ResBlock(channels, emb_channels, dropout=dropout),
        ResBlock(channels, emb_channels, dropout=dropout),
        Downsample(channels),
    ])
  def forward(self, x, emb):
    cache = []
    for layer in self.seq:
      x = layer(x, emb)
      cache += [x]
    return x, cache

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
  def __init__(self, channels, emb_channels, dropout=0.0):
    super().__init__()
    # on the up, bundle resnets with upsampling so upsampling can be simpler
    self.seq = nn.ModuleList([
        TimestepEmbedSequential(ResBlock(2*channels, emb_channels, channels, dropout=dropout), Upsample(channels)),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        TimestepEmbedSequential(ResBlock(2*channels, emb_channels, channels), Upsample(channels)),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
    ])
  def forward(self, x, emb, cache):
    cache = cache[::-1]
    for i in range(len(self.seq)):
      layer, hoz_skip = self.seq[i], cache[i]
      x = th.cat([x, hoz_skip], 1)
      x = layer(x, emb)
    return x

class ResBlock(nn.Module):
  def __init__(self, channels, emb_channels, out_channels=None, dropout=0.0):
    super().__init__()
    self.out_channels = out_channels or channels

    # TODO: check out group norm. it's a bit sus
    self.in_layers = nn.Sequential(
        nn.GroupNorm(8, channels),
        nn.SiLU(),
        nn.Conv3d(channels, self.out_channels, (3, 3, 3), padding=(1, 1, 1))
    )
    self.emb_layers = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_channels, self.out_channels)
    )
    self.out_layers = nn.Sequential(
        nn.GroupNorm(8, self.out_channels),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        zero_module(nn.Conv3d(self.out_channels, self.out_channels, (3, 3, 3), padding=(1, 1, 1)))
    )
    if self.out_channels == channels:
      self.skip_connection = nn.Identity()
    else:
      self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) # step down size

  def forward(self, x, emb):
    h = self.in_layers(x)
    emb_out = self.emb_layers(emb)[..., None, None, None]
    h = h + emb_out
    h = self.out_layers(h)
    return self.skip_connection(x) + h
