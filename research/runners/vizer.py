import time
from collections import defaultdict
import pyglet
import copy
from sync_vector_env import SyncVectorEnv
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
from datetime import datetime
from boxLCD.utils import A
import utils
import data
from define_config import env_fn
from pyglet.gl import glClearColor
KEY = pyglet.window.key

def outproc(img):
  return (255*img[...,None].repeat(3, -1).repeat(8, -2).repeat(8,-3)).astype(np.uint8)

class AutoEnv:
  def __init__(self, model, C):
    self.env = env_fn(C)()
    self.window_batch = None
    self.C = C
    self.model = model
    self.tot_count = 0

  def reset(self):
    self.tot_count = 0
    obses = {key: [] for key in self.env.observation_space.spaces}
    obs = self.env.reset()
    for key, val in self.env.reset().items():
      obses[key] += [val]
    acts = []
    for _ in range(9):
      act = self.env.action_space.sample()
      obs = self.env.step(act)[0]
      for key, val in obs.items():
        obses[key] += [val]
      acts += [act]
    obses = {key: np.stack(val, 0)[None] for key, val in obses.items()}
    acts = np.stack(acts, 0)[None]
    self.window_batch = obses
    self.window_batch['acts'] = acts
    img = outproc(obses['lcd'][0,-1])
    self.count = self.window_batch['lcd'].shape[1] - 1
    for key, val in self.window_batch.items():
      bs, lenk, *extra = val.shape
      pad = np.zeros([bs, 200-lenk, *extra])
      self.window_batch[key] = np.concatenate([val, pad], 1)
    return img, img

  def step(self, act):
    self.tot_count += 1
    obs, rew, done, info = self.env.step(act)
    truth = obs['lcd']
    self.window_batch['acts'][:, self.count] = act[None]
    batch = {key: torch.as_tensor(1.0 * val).float().to(self.C.device) for key, val in self.window_batch.items()}
    lcd_shape = batch['lcd'].shape
    batch['lcd'] = batch['lcd'].flatten(-2)
    out = self.model.onestep(batch, self.count)
    batch['lcd'] = out['lcd'].reshape(lcd_shape)
    self.window_batch = {key: val.detach().cpu().numpy() for key, val in batch.items()}
    pred = self.window_batch['lcd'][:,self.count][0]
    if self.count == 198:
      self.window_batch = {key: np.concatenate([val[:,1:], val[:,:1]], axis=1) for key, val in self.window_batch.items()}
      #val = self.window_batch['lcd']
      #self.window_batch['lcd'] = np.concatenate([val[:,1:], val[:,:1]], axis=1)
    self.count = min(1+self.count, 198)
    return outproc(truth), outproc(pred)

  def make_prompt(self):
    pass

class Vizer:
  def __init__(self, C):
    super().__init__(C)
    C.block_size = C.ep_len
    print('wait dataload')
    self.train_ds, self.test_ds = data.load_ds(C)
    print('dataloaded')
    self.writer = SummaryWriter(C.logdir)
    self.logger = utils.dump_logger({}, self.writer, 0, C)
    self.tvenv = SyncVectorEnv([env_fn(C, 0 + i) for i in range(C.num_envs)], C=C)  # test vector env
    self.autoenv = AutoEnv(self.model, C)

    #bigC = copy.deepcopy(C)
    #bigC.lcd_h *= 4
    #bigC.lcd_w *= 4
    # self.big_tvenv = SyncVectorEnv([env_fn(bigC, 0 + i) for i in range(C.num_envs)], C=bigC)  # test vector env
    loadpath = self.C.weightdir / 'weights.pt'
    self.model.load_state_dict(torch.load(loadpath))
    self.window = pyglet.window.Window(1280, 720)
    self.C = C
    self.paused = False
    self.held_down = defaultdict(lambda: 0)
    self.messages = defaultdict(lambda: 0)
    self.render()
    def on_key_press(symbol, modifiers):
      if symbol == KEY.SPACE:
        self.paused = not self.paused
      if symbol == KEY.I:
        import ipdb; ipdb.set_trace()
      if symbol == KEY.ESCAPE:
        exit()
      if symbol == KEY.S:
        self.messages['sample'] = 1
      if symbol == KEY._0:
        self.messages['reset'] = 1
      self.held_down[symbol] = 1
    def on_key_release(symbol, modifiers):
      self.held_down[symbol] = 0
    self.window.set_handlers(on_key_press=on_key_press, on_key_release=on_key_release)

  def check_message(self, str):
      if self.messages[str]:
        self.messages[str] = False; return True
      else:
        return False

  def run(self, commands=[]):
    if isinstance(commands, str):
      self.messages[commands] = True
    else:
      for key in commands:
        self.messages[key] = True  # TODO: some way to clear the buffer of past key presses

    atruth, apred = self.autoenv.reset()
    i = 0
    lcds = []
    while True:
      kwargs = {}
      if self.check_message('sample'):
        acts, lcds = self.sample()
      imgs = []
      k = (i + 1) % self.C.ep_len
      for j in range(len(lcds)):
        imgs += [((2*j,0), lcds[j][k])]
      imgs += [
        ((0, 3), atruth),
        ((2.1, 3), apred),
        ]
      kwargs['imgs'] = imgs
      atruth, apred = self.autoenv.step(self.env.action_space.sample())
      kwargs['texts'] = [((0.5,3.5), self.autoenv.tot_count)]

      if self.check_message('reset'):
        print('reset')
        atruth, apred = self.autoenv.reset()
      
      self.render(**kwargs)
      if not self.paused:
        i = (i + 1) % int(1e9)
      time.sleep(0.01)

  def sample(self):
    print("sampling")
    obses = {key: [] for key in self.env.observation_space.spaces}
    obs = self.env.reset()
    for key, val in self.env.reset().items():
      obses[key] += [val]

    acts = []
    for _ in range(self.C.ep_len - 1):
      act = self.env.action_space.sample()
      obs = self.env.step(act)[0]
      for key, val in obs.items():
        obses[key] += [val]
      acts += [act]
    acts += [np.zeros_like(act)]
    obses = {key: np.stack(val, 0)[None] for key, val in obses.items()}
    acts = np.stack(acts, 0)
    acts = torch.as_tensor(acts, dtype=torch.float32).to(self.C.device)[None]
    prompts = {key: torch.as_tensor(1.0 * val[:, :5]).to(self.C.device) for key, val in obses.items()}
    lcds = []
    lcds += [outproc(self.model.sample(1, cond=acts, prompts=prompts)[0]['lcd'][0, :, 0].cpu().numpy())]
    lcds += [outproc(self.model.sample(1, cond=acts, prompts=prompts)[0]['lcd'][0, :, 0].cpu().numpy())]
    lcds += [outproc(self.model.sample(1, cond=acts, prompts=prompts)[0]['lcd'][0, :, 0].cpu().numpy())]
    lcds += [outproc(self.model.sample(1, cond=acts, prompts=prompts)[0]['lcd'][0, :, 0].cpu().numpy())]
    truth = outproc(obses['lcd'][0])

    return acts, [truth, *lcds]

  def render(self, return_rgb_array=False, texts=[], size=4, imgs=[], action=None):
    # if self._c.dark_mode:
    glClearColor(0.3, 0.3, 0.3, 1)
    # else:
    #    glClearColor(1,1,1,1)
    images = []
    xys = []
    for xy_img in imgs:
      xy, img = xy_img
      xy = np.array(xy) * self.C.env_size
      xys += [xy]
      images += [pyglet.image.ImageData(img.shape[1], img.shape[0], 'RGB', img.tobytes(), pitch=img.shape[1] * -3)]

    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    for i in range(len(images)):
      images[i].blit(*xys[i])
    for xy_text in texts:
      xy, text = xy_text
      xy = np.array(xy) * self.C.env_size
      label = pyglet.text.HTMLLabel(f'<font face="Times New Roman" size="{size}">{text}</font>', x=xy[0], y=xy[1], anchor_x='center', anchor_y='center')
      label.draw()
    arr = None
    if return_rgb_array:
      buffer = pyglet.image.get_buffer_manager().get_color_buffer()
      image_data = buffer.get_image_data()
      arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
      arr = arr.reshape(buffer.height, buffer.width, 4)
      arr = arr[::-1, :, 0:3]
    self.window.flip()
    self.onetime_geoms = []
    # return arr if return_rgb_array else self.isopen
