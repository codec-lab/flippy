import math
from collections import OrderedDict
from typing import Generic, TypeVar

ISCLOSE_RTOL = 1e-5
ISCLOSE_ATOL = 1e-8

def isclose(a, b, *, rtol=ISCLOSE_RTOL, atol=ISCLOSE_ATOL):
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)

def logsumexp(*args):
    max_arg = max(args)
    return max_arg + math.log(sum(math.exp(arg - max_arg) for arg in args))

def softmax_dict(d: dict):
    max_value = max(d.values())
    keys, values = zip(*d.items())
    exp_values = [math.exp(v - max_value) for v in values]
    total = sum(exp_values)
    return {k: v/total for k, v in zip(keys, exp_values)}

KT = TypeVar('KT')
VT = TypeVar('VT')

class LRUCache(Generic[KT, VT]):
    def __init__(self, max_size=1024):
        self.max_size = max_size if max_size is not None else float('inf')
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def clear(self):
        self._cache.clear()

    def __getitem__(self, key):
        val = self._cache.pop(key)
        self._cache[key] = val
        return val

    def __setitem__(self, key, val):
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = val
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def __contains__(self, key: KT) -> bool:
        contains = key in self._cache
        if contains:
            self.hits += 1
        else:
            self.misses += 1
        return contains

    def keys(self):
        return self._cache.keys()

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"LRUCache({self._cache}, max_size={self.max_size})"
