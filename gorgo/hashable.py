from functools import cached_property

"""
Primitive container types that can be hashed.
They are immutable but otherwise behave like normal containers.
The main use case for these is to be used as keys in return distributions
"""

class hashabledict(dict):
    @cached_property
    def _hash(self):
        return hash(tuple(sorted(self.items())))
    def __hash__(self):
        return self._hash
    def __repr__(self) -> str:
        return super().__repr__()
    def _immutable_error(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} is immutable; use {self.__class__.__name__}.copy() first")
    __setitem__ = _immutable_error
    __delitem__ = _immutable_error
    update = _immutable_error
    clear = _immutable_error
    pop = _immutable_error
    popitem = _immutable_error
    setdefault = _immutable_error
    __ior__ = _immutable_error

class hashablelist(list):
    @cached_property
    def _hash(self):
        return hash(tuple(self))
    def __hash__(self):
        return self._hash
    def __repr__(self) -> str:
        return super().__repr__()
    def _immutable_error(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} is immutable; use {self.__class__.__name__}.copy() first")
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

