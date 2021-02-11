import pathlib
import re
import numpy as np

class AttrDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

def args_type(default):
  if isinstance(default, bool): return lambda x: bool(['False', 'True'].index(x))
  if isinstance(default, int): return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
  if isinstance(default, pathlib.Path): return lambda x: pathlib.Path(x).expanduser()
  return type(default)

class X:
  """
  Singleton object to be able to easily create numpy arrays without having to type as much

  usage:
    >>> A = X()
    >>> arr = A[1, 2, 3]
    array([1, 2, 3])
    >>> arr = A[-1.0, 1.0]
    array([-1.0, 1.0])
  """
  def __getitem__(self, stuff):
    return np.array(stuff)
A = X()

class NamedArray():
  """
  Gives an array an interface like a dictionary, so you can index into it by keys.
  Also handles mapping between different ranges, like [-1,1] to the true value bounds.
  This is convenient for ensuring consistent scaling for feeding in to neural networks.

  NamedArray is one way to handle named array elements. Another option is to convert back and forth between vector and dictionary.
  NamedArray can be faster and more convenient.

  #usage:
  #  >>> obs = np.array([10.0, 5.0, 3.2])
  #  >>> obs_info = {'}
  #  >>> named_obs = NamedArray(obs, obs_info, do_map=True)

  args:
    arr (np.ndarray) of shape (..., N)
    arr_info (dict[key: index_bounds]) to map from an index to the true bounds of that value.
    do_map (bool) Whether or not to map between the arr_info bounds or not. when off, no scaling is applied to values.
  """
  def __init__(self, arr, arr_info, do_map=True):
    self.arr = arr
    self.arr_info = arr_info
    self.do_map = do_map

  def _name2idx(self, name):
    """get the index of an element in the array"""
    return list(self.arr_info.keys()).index(name)

  def todict(self):
    d = {}
    for key in self.arr_info:
      d[key] = self[key]
    return d

  def __call__(self, key):
    """support parenthes as well"""
    return self[key]

  def __getitem__(self, key):
    """access an array by index name or list of names"""
    if isinstance(key, str):
      idx = self._name2idx(key)
      if self.do_map:
        return mapto(self.arr[..., idx], self.arr_info[key])
    elif isinstance(key, list):
      idx = [self._name2idx(subk) for subk in key]
      if self.do_map:
        bounds = np.array([self.arr_info[subk] for subk in key]).T
        return mapto(self.arr[..., idx], bounds)
    else:
      raise NotImplementedError
    return self.arr[..., idx]

  def __setitem__(self, key, item):
    """set the array indexes by name or list of names"""
    if isinstance(key, str):
      idx = self._name2idx(key)
      if self.do_map:
        self.arr[..., idx] = rmapto(item, self.arr_info[key])
        return
    elif isinstance(key, list):
      idx = [self._name2idx(subk) for subk in key]
      if self.do_map:
        bounds = np.array([self.arr_info[subk] for subk in key]).T
        self.arr[..., idx] = rmapto(item, bounds)
        return
    else:
      raise NotImplementedError
    self.arr[..., idx] = item

# general dictionary and list utils
def subdict(dict, subkeys): return {key: dict[key] for key in subkeys}
def sortdict(x): return subdict(x, sorted(x))
def subdlist(dict, subkeys): return [dict[key] for key in subkeys]
# filter or negative filter
def filtdict(dict, phrase): return {key: dict[key] for key in dict if re.match(phrase, key) is not None}
def nfiltdict(dict, phrase): return {key: dict[key] for key in dict if re.match(phrase, key) is None}
def filtlist(list, phrase): return [item for item in list if re.match(phrase, item) is not None]
def nfiltlist(list, phrase): return [item for item in list if re.match(phrase, item) is None]

# env specific stuff
def get_angle(sin, cos): return np.arctan2(sin, cos)
def make_rot(angle): return A[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
# map from -1,1 to bounds
def mapto(a, lowhigh): return ((a + 1.0) / (2.0) * (lowhigh[1] - lowhigh[0])) + lowhigh[0]
# map from bounds to -1,1
def rmapto(a, lowhigh): return ((a - lowhigh[0]) / (lowhigh[1] - lowhigh[0]) * (2)) + -1


from pyglet.gl import glClearColor
import pyglet
set_width = False
# this is a really bad idea if the underlying code changes.
# i really should make my own rendering class to copy the gym.envs.classic_control.rendering code.
def monkey_patch_render(self, return_rgb_array=False, lcd=None):
    global set_width
    glClearColor(1,1,1,1)
    if not set_width:
      self.window.width *= 2
      set_width = True
    if lcd is not None:
      dw = (self.window.width/2) / 30.0
      img = pyglet.image.ImageData(lcd.shape[1], lcd.shape[0], 'RGB', lcd.tobytes(), pitch=lcd.shape[1]*-3)
      h, w = A[lcd.shape[:2]]/30.0
      self.draw_line((dw, 0), (dw,h*2), color=(0,0,0)) # right boundary
      #self.draw_line((0,0), (0,1,), color=(1,1,1)) # white line on bottom. fixes image gets tinted by last color used
      self.draw_line((dw, 0), (dw, h), color=(0,0,0))
      self.draw_line((dw, h), (dw+w, h), color=(0,0,0))
      self.draw_line((dw+w, 0), (dw+w, h), color=(0,0,0))
      self.draw_line((0, 0), (0, 1), color=(1,1,1))
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    if lcd is not None:
      img.blit(self.window.width//2,0)

    self.transform.enable()
    for geom in self.geoms:
        geom.render()
    for geom in self.onetime_geoms:
        geom.render()
    self.transform.disable()
    arr = None
    if return_rgb_array:
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        # In https://github.com/openai/gym-http-api/issues/2, we
        # discovered that someone using Xmonad on Arch was having
        # a window of size 598 x 398, though a 600 x 400 window
        # was requested. (Guess Xmonad was preserving a pixel for
        # the boundary.) So we use the buffer height/width rather
        # than the requested one.
        arr = arr.reshape(buffer.height, buffer.width, 4)
        arr = arr[::-1,:,0:3]
    self.window.flip()
    self.onetime_geoms = []
    return arr if return_rgb_array else self.isopen