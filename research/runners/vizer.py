import time
from collections import defaultdict
import pyglet
import copy
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
import torch as th
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
from datetime import datetime
from boxLCD.utils import A, NamedArray
import utils
import data
from define_config import env_fn
from pyglet.gl import glClearColor
KEY = pyglet.window.key

def outproc(img):
  return (255 * img[..., None].repeat(3, -1).repeat(8, -2).repeat(8, -3)).astype(np.uint8)

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
    for key, val in obs.items():
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
    img = outproc(obses['lcd'][0, -1])
    self.count = self.window_batch['lcd'].shape[1] - 1
    for key, val in self.window_batch.items():
      bs, lenk, *extra = val.shape
      pad = np.zeros([bs, self.C.window - lenk, *extra])
      self.window_batch[key] = np.concatenate([val, pad], 1)
    return img, img

  def step(self, act):
    self.tot_count += 1
    obs, rew, done, info = self.env.step(act)
    truth = obs['lcd']
    self.window_batch['acts'][:, self.count] = act[None]
    batch = {key: th.as_tensor(1.0 * val).float().to(self.C.device) for key, val in self.window_batch.items()}
    lcd_shape = batch['lcd'].shape
    batch['lcd'] = batch['lcd'].flatten(-2)
    out = self.model.onestep(batch, self.count, temp=0.1)
    batch['lcd'] = out['lcd'].reshape(lcd_shape)
    self.window_batch = {key: val.detach().cpu().numpy() for key, val in batch.items()}
    pred = self.window_batch['lcd'][:, self.count][0]
    if self.count == self.C.window - 2:
      self.window_batch = {key: np.concatenate([val[:, 1:], val[:, :1]], axis=1) for key, val in self.window_batch.items()}
      #val = self.window_batch['lcd']
      #self.window_batch['lcd'] = np.concatenate([val[:,1:], val[:,:1]], axis=1)
    self.count = min(1 + self.count, self.C.window - 2)
    return outproc(truth), outproc(pred)

  def make_prompt(self):
    pass

class Vizer:
  def __init__(self, model, env, C):
    super().__init__()
    print('wait dataload')
    self.train_ds, self.test_ds = data.load_ds(C)
    print('dataloaded')
    self.model = model
    self.env = env
    self.autoenv = AutoEnv(self.model, C)
    self.model.load(C.weightdir)
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
      if symbol == KEY._0 or symbol == KEY.NUM_0:
        self.messages['reset'] = 1
      if symbol == KEY.R:
        print('RELOAD WEIGHTS')
        self.model.load(C.weightdir)
      if symbol == KEY.G:
        self.messages['goal'] = 1
      self.held_down[symbol] = 1

    def on_key_release(symbol, modifiers):
      self.held_down[symbol] = 0
    self.window.set_handlers(on_key_press=on_key_press, on_key_release=on_key_release)

  def check_message(self, str):
    if self.messages[str]:
      self.messages[str] = False; return True
    else:
      return False

  def opt_through(self, o, a, g):
    o = th.as_tensor(o, dtype=th.float32).to(self.H.device)
    a = th.as_tensor(a, dtype=th.float32).to(self.H.device)
    g = th.as_tensor(g, dtype=th.float32).to(self.H.device)
    og = g.data.clone().detach()
    g.requires_grad = True
    coords = th.as_tensor(self.gxy_coords, dtype=th.long).to(self.H.device)

    for i in range(5):
      loss = -self.ac.q1(o, a, g).mean()
      loss.backward()
      print(f'{i} loss {loss}')
      g.grad[..., coords] = 0.0
      g.data.add_(-0.0001 * g.grad)
      g.data.clamp_(-1, 1)
      g.grad[:] = 0.0
    print('delta', th.linalg.norm(g - og))
    return g.detach().cpu().numpy()

  def look_ahead(self, prompto, a):
    sample = self.model.sample(1, acts=a, prompts={'lcd': prompto})[0]
    return sample['lcd']

  def sample_traj(self, prompto, prompta, g):
    prompto = th.as_tensor(1.0 * np.stack(prompto), dtype=th.float32).to(self.C.device)[None]
    prompta = 1.0 * np.stack(prompta)
    g = th.as_tensor(1.0 * g, dtype=th.float32).to(self.C.device)[None]
    N, A = prompta.shape
    extra = np.random.uniform(-1, 1, size=(self.C.window - N, A))
    a = np.concatenate([prompta, extra])
    a = th.as_tensor(a, dtype=th.float32).to(self.C.device)[None]
    oa = a.data.clone().detach()
    a.requires_grad = True
    for i in range(10):
      o = self.look_ahead(prompto, a)
      out = self.model.forward({'lcd': o, 'acts':a})
      loss = -self.model.dist_head(out).log_prob(g.flatten(-2)).mean()
      loss.backward()
      a.grad[:,:10] = 0.0
      a.data.add_(-a.grad)
      a.data.clamp_(-1,1)
      a.grad[:] = 0.05
      #print(a)
    print('delta', ((a-oa)**2).mean())
    return o.detach().cpu().numpy()

  def do_goal(self):
    eobs = self.env.reset()
    tenv = env_fn(self.C)()
    obs = tenv.reset()
    obses = [obs['lcd']]
    acts = []
    for i in range(10):
      act = tenv.action_space.sample()
      acts += [act]
      obs, rew, done, info = tenv.step(act)
      obses += [obs['lcd']]
    obses = obses[:-1]
    xkeys = [x for x in self.env.obs_keys if 'x:p' in x]
    neobs = NamedArray(eobs['full_state'], self.env.obs_info, do_map=False)

    while True:
      kwargs = {}
      if self.held_down[KEY.LEFT]:
        neobs[xkeys] -= 0.05
      if self.held_down[KEY.RIGHT]:
        neobs[xkeys] += 0.05
      self.env.reset(full_state=neobs.arr)
      goal_lcd = self.env.lcd_render()
      kwargs['imgs'] = [((1, 1), outproc(goal_lcd))]
      #kwargs['imgs'] = [((1, 1), self.env.lcd_render(height=128, width=int(self.C.wh_ratio * 128), pretty=True))]
      self.render(**kwargs)

      if self.held_down[KEY.SPACE]:
        traj = self.sample_traj(obses, acts, goal_lcd)[0,:,0]
        for i in range(len(traj)):
          kwargs['imgs'] = [((1, 1), outproc(traj[i]))]
          #kwargs['imgs'] = [((1, 1), self.env.lcd_render(height=128, width=int(self.C.wh_ratio * 128), pretty=True))]
          self.render(**kwargs)



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
      k = (i + 1) % self.C.window
      for j in range(len(lcds)):
        imgs += [((2 * j, 0), lcds[j][k])]
      imgs += [
          ((0, 3), atruth),
          ((2.1, 3), apred),
      ]
      kwargs['imgs'] = imgs
      if not self.paused:
        atruth, apred = self.autoenv.step(self.env.action_space.sample())
      kwargs['texts'] = [((0.5, 3.5), f'truth. {self.autoenv.tot_count}')]

      if self.check_message('reset'):
        print('reset')
        atruth, apred = self.autoenv.reset()

      if self.check_message('goal'):
        self.do_goal()

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
    for _ in range(self.C.window - 1):
      act = self.env.action_space.sample()
      obs = self.env.step(act)[0]
      for key, val in obs.items():
        obses[key] += [val]
      acts += [act]
    acts += [np.zeros_like(act)]
    obses = {key: np.stack(val, 0)[None] for key, val in obses.items()}
    acts = np.stack(acts, 0)
    acts = th.as_tensor(acts, dtype=th.float32).to(self.C.device)[None]
    prompts = {key: th.as_tensor(1.0 * val[:, :5]).to(self.C.device) for key, val in obses.items()}
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
      xy = np.array(xy) * self.C.lcd_base * self.C.wh_ratio * 8
      xys += [xy]
      images += [pyglet.image.ImageData(img.shape[1], img.shape[0], 'RGB', img.tobytes(), pitch=img.shape[1] * -3)]

    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    for i in range(len(images)):
      images[i].blit(*xys[i])
    for xy_text in texts:
      xy, text = xy_text
      xy = np.array(xy) * self.C.lcd_base * self.C.wh_ratio * 8
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
