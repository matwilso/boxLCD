import re
import numpy as np

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

class WrappedArray():
  """
  Gives an array an interface like a dictionary, so you can index into it by keys.
  Also handles mapping between different ranges, like [-1,1] to the true value bounds.
  This is convenient for ensuring consistent scaling for feeding in to neural networks.

  WrappedArray is one way to handle named array elements. Another option is to convert back and forth between vector and dictionary.
  WrappedArray can be faster and more convenient.

  #usage:
  #  >>> obs = np.array([10.0, 5.0, 3.2])
  #  >>> obs_info = {'}
  #  >>> wrapped_obs = WrappedArray(obs, obs_info, do_map=True)

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
    return self.arr_info['keys'].index(name)

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