import copy
from research.nets.video_models.gaussian_diffusion import GaussianDiffusion
from research import utils
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torch as th
from torch import distributions as thd
from torch import nn
import torch.nn.functional as F
from research.nets.common import SelfAttention, TimestepEmbedSequential, ResBlock, timestep_embedding, zero_module, TransformerBlock
from ._base import VideoModel
from einops import rearrange, repeat, parse_shape
from jax.tree_util import tree_map

class DiffusionVideo(VideoModel):
  def __init__(self, env, G):
    super().__init__(env, G)
    G.dropout = 0.0
    G.timesteps = 500 # seems to work pretty well for MNIST
    G.timestep_embed = 64
    state_n = env.observation_space.spaces['proprio'].shape[0]
    self.embed_size = G.n_embed
    self.net = SimpleUnet3d(G)
    self.diffusion = GaussianDiffusion(G.timesteps)
    self._init()

  def evaluate(self, epoch, writer, batch, arbiter=None):
    metrics = {}

    if run_sampling := False:
      n = batch['lcd'].shape[0]
      out = self.sample(n, prompts=batch, prompt_n=self.G.prompt_n)
      #self._diffusion_video(epoch, writer, out['diffusion_sampling'], name='diffusion_sampling', prompt_n=None)
      self._diffusion_video(epoch, writer, out['diffusion_pred'], name='diffusion_pred', prompt_n=None)

    self._prompted_eval(epoch, writer, metrics, batch, arbiter, make_video=self.G.make_video)
    metrics = tree_map(lambda x: th.as_tensor(x).cpu(), metrics)
    return metrics

  def _diffusion_video(self, epoch, writer, pred, truth=None, name=None, prompt_n=None):
    pred = pred[:,:4]
    pred = th.cat([pred, th.zeros_like(pred[:,:,:,:,:1])], axis=4)
    pred = th.cat([pred, th.zeros_like(pred[:,:,:,:,:,:1])], axis=5)
    out = rearrange(pred, 's bs n t h w -> n s (bs h) (t w)')
    out = repeat(out, 'n s h w -> n s c h w', c=3)
    out = repeat(out, 'n s c h w -> n s c (h h2) (w w2)', h2=2, w2=2)
    utils.add_video(writer, name, out, epoch, fps=60)
    writer.flush()
    print('FLUSH')


  def loss(self, batch):
    lcd = (batch['lcd'][:,None] * 2) - 1
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
    samps = [al['sample'] for al in all_samples]
    preds = [al['pred_xstart'] for al in all_samples]
    diffusion_sampling = th.stack(samps)
    diffusion_pred = th.stack(preds)
    raw_samples = all_samples[-1]['sample']
    raw_samples = (raw_samples + 1) / 2
    return {'lcd': rearrange(raw_samples, 'bs c t h w -> bs t c h w'), 'diffusion_sampling': diffusion_sampling, 'diffusion_pred': diffusion_pred}


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
    self.down = Down(G.window, channels, time_embed_dim, dropout=dropout)
    self.turn = ResBlock(channels, time_embed_dim, dropout=dropout)
    self.up = Up(G.window, channels, time_embed_dim, dropout=dropout)
    self.out = nn.Sequential(
      nn.GroupNorm(16, channels),
      nn.SiLU(),
      nn.Conv3d(channels, 2, (3, 3, 3), padding=(1, 1, 1)),
    )
    self.G = G

  def forward(self, x, timesteps):
    emb = self.time_embed(timestep_embedding(timesteps.float(), self.G.timestep_embed, timesteps.shape[0]))
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
    self.conv = nn.Conv3d(channels, out_channels, kernel_size=(3, 3, 3), stride=(1, stride, stride), padding=(1, 1, 1))
    #self.conv = nn.Conv3d(channels, out_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1))

  def forward(self, x, emb=None):
    return self.conv(x)

class Down(nn.Module):
  def __init__(self, window, channels, emb_channels, dropout=0.0):
    super().__init__()
    self.seq = nn.ModuleList([
        Downsample(1, channels, 1),  # not really a downsample, just makes the code simpler to share
        #ResBlock(channels, emb_channels, dropout=dropout),
        ResBlock(channels, emb_channels, dropout=dropout),
        TimestepEmbedSequential(ResBlock(channels, emb_channels, channels, dropout=dropout), AttentionBlock(window, channels)),
        Downsample(channels),
        #ResBlock(channels, emb_channels, dropout=dropout),
        TimestepEmbedSequential(ResBlock(channels, emb_channels, channels, dropout=dropout), AttentionBlock(window, channels)),
        ResBlock(channels, emb_channels, dropout=dropout),
        Downsample(channels),
    ])
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
    self.seq = nn.ModuleList([
        TimestepEmbedSequential(ResBlock(2*channels, emb_channels, channels, dropout=dropout), Upsample(channels)),
        #ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        TimestepEmbedSequential(ResBlock(2*channels, emb_channels, channels, dropout=dropout), AttentionBlock(window, channels)),
        TimestepEmbedSequential(ResBlock(2*channels, emb_channels, channels), Upsample(channels)),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        #ResBlock(2*channels, emb_channels, channels, dropout=dropout),
        TimestepEmbedSequential(ResBlock(2*channels, emb_channels, channels, dropout=dropout), AttentionBlock(window, channels)),
        ResBlock(2*channels, emb_channels, channels, dropout=dropout),
    ])
  def forward(self, x, emb, pass_through):
    pass_through = pass_through[::-1]
    for i in range(len(self.seq)):
      layer, hoz_skip = self.seq[i], pass_through[i]
      x = th.cat([x, hoz_skip], 1)
      x = layer(x, emb)
    return x


class AttentionBlock(nn.Module):
  def __init__(self, n, n_embed):
    super().__init__()
    n_head = n_embed // 8
    self.attn = TransformerBlock(n, n_embed, n_head, causal=False)
    #self.attn = SelfAttention(n, n_embed, n_head, causal=False)
    self.register_buffer("time_embed", timestep_embedding(th.linspace(0, 1, n), n_embed, max_period=1).T)

  def forward(self, x, emb=None):
    x = x + self.time_embed[None,:,:,None,None]
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
        nn.Conv3d(channels, self.out_channels, (3, 3, 3), padding=(1, 1, 1))
    )
    self.emb_layers = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_channels, self.out_channels)
    )
    self.out_layers = nn.Sequential(
        nn.GroupNorm(16, self.out_channels),
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
