import torch
import scipy
import re
import numpy as np
from tensorflow import nest

# utils
def subdict(dict, keys): return {key: dict[key] for key in keys}
def filter(dict, name): return {key: dict[key] for key in dict if re.match(name, key) is not None}
def lfilter(list, name): return [item for item in list if re.match(name, item) is not None]
def sortdict(x): return subdict(x, sorted(x))
def subdlist(dict, keys): return [dict[key] for key in keys]

def get_angle(sin, cos): return np.arctan2(sin, cos)
# map from -1,1 to bounds
def mapto(a, lowhigh): return ((a + 1.0) / (2.0) * (lowhigh[1] - lowhigh[0])) + lowhigh[0]
# map from bounds to -1,1
def rmapto(a, lowhigh): return ( (a - lowhigh[0]) / (lowhigh[1]-lowhigh[0]) * (2)) + -1
def umapto(a, from_lh, to_lh): return ( (a - from_lh[0]) / (from_lh[1]-from_lh[0]) * (to_lh[0]+to_lh[1])) + to_lh[0]

def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert reward.ndim == value.ndim, (reward.shape, value.shape)
    dims = list(range(reward.ndim))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = torch.permute(reward, dims)
        value = torch.permute(value, dims)
        pcont = torch.permute(pcont, dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan(lambda agg, cur: cur[0] + cur[1] * lambda_ * agg, (inputs, pcont), bootstrap, reverse=True)
    if axis != 0:
        returns = torch.permute(returns, dims)
    return returns


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


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

def make_rot(angle):
    return A[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class DWrap():
    """
    Dictionary Wrapper

    Gives an array an interface like a dictionary, so you can index into it by keys.
    Also handles mapping between different ranges, like [-1,1] to true bounds.

    This is one way to handle named array elements. The other way is just convert back and forth between vector and dictionary.
    This way can be faster and more convenient.
    """

    def __init__(self, arr, arr_info, map=True):
        self.arr = arr
        self.arr_info = arr_info
        self.map = map

    def _name2idx(self, name):
        """get the index of an element in the array"""
        return self.arr_info['keys'].index(name)

    def __getitem__(self, key):
        """access an array by index name or list of names"""
        if isinstance(key, str):
            idx = self._name2idx(key)
            if self.map:
                return mapto(self.arr[..., idx], self.arr_info[key])
        else:
            idx = [self._name2idx(subk) for subk in key]
            if self.map:
                bounds = np.array([self.arr_info[subk] for subk in key]).T
                return mapto(self.arr[..., idx], bounds)
        return self.arr[..., idx]

    def __setitem__(self, key, item):
        """set the array indexes by name or list of names"""
        if isinstance(key, str):
            idx = self._name2idx(key)
            if self.map:
                self.arr[..., idx] = rmapto(item, self.arr_info[key])
                return
        else:
            idx = [self._name2idx(subk) for subk in key]
            if self.map:
                bounds = np.array([self.arr_info[subk] for subk in key]).T
                self.arr[..., idx] = rmapto(item, bounds)
                return
        self.arr[..., idx] = item


def static_scan(fn, inputs, start, reverse=False):
    last = start
    outputs = [[] for _ in nest.flatten(start)]
    indices = range(len(nest.flatten(inputs)[0]))
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = nest.map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [torch.stack(x, 0) for x in outputs]
    return nest.pack_sequence_as(start, outputs)
