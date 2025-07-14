import collections.abc
from functools import cached_property

"""
Primitive container types that can be hashed.
They are immutable but otherwise behave like normal containers.
The main use case for these is to be used as keys in return distributions
"""

def make_hashable(obj):
    if isinstance(obj, dict):
        return hashabledict(obj)
    elif isinstance(obj, list):
        return hashablelist(obj)
    elif isinstance(obj, set):
        return hashableset(obj)
    elif isinstance(obj, (collections.abc.ItemsView, collections.abc.KeysView)):
        return hashableset(obj)
    elif isinstance(obj, collections.abc.ValuesView):
        return hashablelist(obj)
    return obj

class hashabledict(dict):
    @cached_property
    def _hash(self):
        try:
            return hash(frozenset(self.items()))
        except TypeError:
            return self._recursive_hash()
    def _recursive_hash(self):
        # This only handles cases where the immediate values that are containers
        # can be coerced into hashable versions
        items = ((k, make_hashable(v)) for k, v in self.items())
        return hash(frozenset(items))
    def __hash__(self):
        return self._hash
    def __repr__(self) -> str:
        return super().__repr__()
    def _immutable_error(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} is immutable; use {self.__class__.__name__}.copy() first")
    def __or__(self, *args, **kwargs):
        return hashabledict(super().__or__(*args, **kwargs))
    def __ror__(self, *args, **kwargs):
        return hashabledict(super().__ror__(*args, **kwargs))
    __setitem__ = _immutable_error
    __delitem__ = _immutable_error
    update = _immutable_error
    clear = _immutable_error
    pop = _immutable_error
    popitem = _immutable_error
    setdefault = _immutable_error
    __ior__ = _immutable_error
    def __reduce__(self):
        return (
            hashabledict,
            (dict(self),)
        )

class hashablelist(list):
    @cached_property
    def _hash(self):
        try:
            return hash(tuple(self))
        except TypeError:
            return self._recursive_hash()
    def _recursive_hash(self):
        # This only handles cases where the immediate values that are containers
        # can be coerced into hashable versions
        items = (make_hashable(v) for v in self)
        return hash(tuple(items))
    def __hash__(self):
        return self._hash
    def __repr__(self) -> str:
        return super().__repr__()
    def _immutable_error(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} is immutable; use {self.__class__.__name__}.copy() first")
    def __add__(self, *args, **kwargs):
        return hashablelist(super().__add__(*args, **kwargs))
    def __radd__(self, other):
        return self.__add__(other)
    def __mul__(self, *args, **kwargs):
        return hashablelist(super().__mul__(*args, **kwargs))
    def __rmul__(self, other):
        return self.__mul__(other)
    def __getitem__(self, *args, **kwargs):
        if isinstance(args[0], slice):
            return hashablelist(super().__getitem__(*args, **kwargs))
        return super().__getitem__(*args, **kwargs)
    __setitem__ = _immutable_error
    __delitem__ = _immutable_error
    append = _immutable_error
    extend = _immutable_error
    insert = _immutable_error
    pop = _immutable_error
    remove = _immutable_error
    clear = _immutable_error
    __iadd__ = _immutable_error
    sort = _immutable_error
    __imul__ = _immutable_error
    reverse = _immutable_error
    def __reduce__(self):
        return (
            hashablelist,
            (list(self),)
        )

class hashableset(set):
    @cached_property
    def _hash(self):
        return hash(frozenset(self))
    def __hash__(self):
        return self._hash
    def __repr__(self) -> str:
        return repr(set(self))
    def _immutable_error(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} is immutable; use {self.__class__.__name__}.copy() first")
    def __or__(self, *args, **kwargs):
        return hashableset(super().__or__(*args, **kwargs))
    def __ror__(self, *args, **kwargs):
        return hashableset(super().__ror__(*args, **kwargs))
    def __and__(self, *args, **kwargs):
        return hashableset(super().__and__(*args, **kwargs))
    def __rand__(self, *args, **kwargs):
        return hashableset(super().__rand__(*args, **kwargs))
    update = _immutable_error
    intersection_update = _immutable_error
    __ior__ = _immutable_error
    __iand__ = _immutable_error
    __ixor__ = _immutable_error
    difference_update = _immutable_error
    symmetric_difference_update = _immutable_error
    add = _immutable_error
    remove = _immutable_error
    discard = _immutable_error
    pop = _immutable_error
    clear = _immutable_error
    def __reduce__(self):
        return (
            hashableset,
            (set(self),)
        )

