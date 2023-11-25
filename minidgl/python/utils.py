"""Utility module."""
from __future__ import absolute_import, division
import sys
from collections.abc import  Iterable
from collections.abc import Mapping
from functools import wraps
import numpy as np

sys.path.append('../python')
import needle as ndl
import needle.backend_ndarray as bn

class Index(object):
    """Index class that can be easily converted to list/tensor.
    minidgl支持list格式和slice格式的data索引。
    Index数据可以转换为NDarray、numpy
    """
    def __init__(self, data):
        self._initialize_data(data)

    def _initialize_data(self, data):
        self.list_data = None
        self.slice_data = None
        self.np_data = None
        self.minidgl_tensor = None
        self._dispatch(data)

    def __iter__(self):
        for i in self.tonumpy():
            yield int(i)

    def __len__(self):
        if self.slice_data is not None and isinstance(self.slice_data,slice):
            slc = self.slice_data
            if slc.step is None:
                return slc.stop-slc.start
            else:
                return (slc.stop-slc.start)//slc.step
        else:
            return len(self.list_data)

    def __getitem__(self, i):
        return int(self.tonumpy()[i])

    def _dispatch(self, data):
        """Store data based on its type."""
        if isinstance(data,np.ndarray):
            self.np_data = data
        elif isinstance(data,list):
            self.list_data = data
        elif isinstance(data,slice):
            self.slice_data = data
        self.tonumpy()
        self.todgltensor()

    def tonumpy(self):
        """Convert to a numpy ndarray."""
        if self.np_data is not None:
            return self.np_data
        elif self.slice_data is not None:
            #convert slice to numpy array
            slc = self.slice_data
            self.np_data = np.arange(slc.start,slc.stop,slc.step).astype(np.int64)
        elif self.list_data is not None:
            print(self.list_data)
            self.np_data = np.array(self.list_data)
            print(self.np_data)
        return self.np_data

    def todgltensor(self,ctx=bn.default_device()):
        """Convert to needle.NDArray.

        convert to numpy array first then to NDArray.
        use ctx as backend device
        """
        if self.minidgl_tensor is None:
            # zero copy from user tensor
            tsor = self.tonumpy()
            dl = ndl.Tensor(array=tsor,device=ctx,dtype="float32")
            self.minidgl_tensor = dl
        return self.minidgl_tensor

    def is_slice(self, start, stop, step=None):
        return (isinstance(self.slice_data, slice)
                and self.slice_data == slice(start, stop, step))

    def __getstate__(self):
        return self.todgltensor()

    def __setstate__(self, state):
        self._initialize_data(state)

def toindex(x):
    return x if isinstance(x, Index) else Index(x)

class LazyDict(Mapping):
    """A readonly dictionary that does not materialize the storage."""
    def __init__(self, fn, keys):
        self._fn = fn
        self._keys = keys

    def __getitem__(self, key):
        if not key in self._keys:
            raise KeyError(key)
        return self._fn(key)

    def __contains__(self, key):
        return key in self._keys

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys

class HybridDict(Mapping):
    """A readonly dictonary that merges several dict-like (python dict, LazyDict).
       If there are duplicate keys, early keys have priority over latter ones
    """
    def __init__(self, *dict_like_list):
        self._dict_like_list = dict_like_list
        self._keys = set()
        for d in dict_like_list:
            self._keys.update(d.keys())

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        for d in self._dict_like_list:
            if key in d:
                return d[key]
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

class ReadOnlyDict(Mapping):
    """A readonly dictionary wrapper."""
    def __init__(self, dict_like):
        self._dict_like = dict_like

    def keys(self):
        return self._dict_like.keys()

    def __getitem__(self, key):
        return self._dict_like[key]

    def __contains__(self, key):
        return key in self._dict_like

    def __iter__(self):
        return iter(self._dict_like)

    def __len__(self):
        return len(self._dict_like)

def build_relabel_map(x, sorted=False):
    """Relabel the input ids to continuous ids that starts from zero.

    Ids are assigned new ids according to their ascending order.

    Only receive numpy data.

    Examples
    --------
    >>> x = [1, 5, 3, 6]
    >>> n2o, o2n = build_relabel_map(x)
    >>> n2o
    [1, 3, 5, 6]
    >>> o2n
    [n/a, 0, n/a, 2, n/a, 3, 4]

    "n/a" will be filled with 0

    Parameters
    ----------
    x : Index
        The input ids.
    sorted : bool, default=False
        Whether the input has already been unique and sorted.

    Returns
    -------
    new_to_old : tensor
        The mapping from new id to old id.
    old_to_new : tensor
        The mapping from old id to new id. It is a vector of length MAX(x).
        One can use advanced indexing to convert an old id tensor to a
        new id tensor: new_id = old_to_new[old_id]
    """
    x = x.tonumpy()
    if not sorted:
        unique_x, _ = np.unique(x)
    else:
        unique_x = x
    map_len = int(np.max(a=unique_x,axis=0)) + 1
    old_to_new = np.zeros(shape=(map_len,),dtype=np.int64)
    old_to_new[unique_x] = np.arange(0,len(unique_x))
    return unique_x, old_to_new

def build_relabel_dict(x):
    """Relabel the input ids to continuous ids that starts from zero.

    The new id follows the order of the given node id list.

    Parameters
    ----------
    x : list
      The input ids.

    Returns
    -------
    relabel_dict : dict
      Dict from old id to new id.
    """
    relabel_dict = {}
    for i, v in enumerate(x):
        relabel_dict[v] = i
    return relabel_dict

class CtxCachedObject(object):
    """A wrapper to cache object generated by different context.

    Note: such wrapper may incur significant overhead if the wrapped object is very light.

    Parameters
    ----------
    generator : callable
        A callable function that can create the object given ctx as the only argument.
    """
    def __init__(self, generator):
        self._generator = generator
        self._ctx_dict = {}

    def get(self, ctx):
        if not ctx in self._ctx_dict:
            self._ctx_dict[ctx] = self._generator(ctx)
        return self._ctx_dict[ctx]

def ctx_cached_member(func):
    """Convenient class member function wrapper to cache the function result.

    The wrapped function must only have two arguments: `self` and `ctx`. The former is the
    class object and the later is the context. It will check whether the class object is
    freezed (by checking the `_freeze` member). If yes, it caches the function result in
    the field prefixed by '_CACHED_' before the function name.
    """
    cache_name = '_CACHED_' + func.__name__
    @wraps(func)
    def wrapper(self, ctx):
        if self._freeze:
            # cache
            if getattr(self, cache_name, None) is None:
                bind_func = lambda _ctx : func(self, _ctx)
                setattr(self, cache_name, CtxCachedObject(bind_func))
            return getattr(self, cache_name).get(ctx)
        else:
            return func(self, ctx)
    return wrapper

def cached_member(func):
    cache_name = '_CACHED_' + func.__name__
    @wraps(func)
    def wrapper(self):
        if self._freeze:
            # cache
            if getattr(self, cache_name, None) is None:
                setattr(self, cache_name, func(self))
            return getattr(self, cache_name)
        else:
            return func(self)
    return wrapper

def is_dict_like(obj):
    return isinstance(obj, Mapping)

def reorder(dict_like, index):
    """Reorder each column in the dict according to the index.

    Parameters
    ----------
    dict_like : dict of tensors
        The dict to be reordered.
    index : dgl.utils.Index
        The reorder index.
    """
    new_dict = {}
    for key, val in dict_like.items():
        idx_ctx = index.tonumpy()
        new_dict[key] = val[idx_ctx]
    return new_dict

def reorder_index(idx, order):
    """Reorder the idx according to the given order

    Parameters
    ----------
    idx : utils.Index
        The index to be reordered.
    order : utils.Index
        The order to follow.
    """
    idx = idx.tonumpy()
    order = order.tonumpy()
    new_idx = idx[order]
    return toindex(new_idx)

def is_iterable(obj):
    """Return true if the object is an iterable."""
    return isinstance(obj, Iterable)
