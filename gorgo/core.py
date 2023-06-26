from typing import Any, Callable, Hashable, Tuple
from gorgo.distributions import Distribution
from gorgo.funcutils import cached_property

############################################
#  Program State
############################################

from collections import namedtuple
StackFrame = namedtuple("StackFrame", "func_src lineno locals")

# a continuation is a function that takes a value and returns a thunk
Continuation = Callable[..., Callable[[], Any]]

class ProgramState:
    def __init__(
        self,
        continuation : Continuation = None,
        name: Hashable = None,
        stack: Tuple[StackFrame] = None,
    ):
        self.continuation = continuation
        self._name = name
        self.stack = stack
        self.init_global_store = GlobalStore()

    def step(self, *args, **kws):
        next_ = self.continuation(*args, **kws)
        global_store = self.init_global_store.copy()
        # HACK: Is there no other reasonable way to access interpreter?
        _cps = next_.__globals__['_cps']
        _cps.global_store_proxy.proxied = global_store
        while True:
            if callable(next_):
                next_ = next_()
            elif isinstance(next_, ProgramState):
                next_.init_global_store = global_store
                return next_
            else:
                raise TypeError(f"Unknown type {type(next_)}")

    @cached_property
    def name(self):
        if self._name is not None:
            return self._name
        if self.stack is None:
            return None
        return tuple((frame.func_src, frame.lineno) for frame in self.stack)

class ReadOnlyProxy(object):
    def __init__(self):
        self.proxied = None
    def __getattr__(self, name):
        if self.proxied is None:
            raise NotImplementedError("Proxying to None")
        return getattr(self.proxied, name)
    def __contains__(self, key):
        if self.proxied is None:
            raise NotImplementedError("Proxying to None")
        return key in self.proxied

class GlobalStore:
    def __init__(self, initial : dict = None):
        self.store = initial if initial is not None else {}

    def copy(self):
        return GlobalStore({**self.store})

    def get(self, key : Hashable, default : Any = None):
        return self.store.get(key, default)

    def __getitem__(self, key : Hashable):
        return self.store[key]

    def __setitem__(self, key : Hashable, value : Any):
        self.store[key] = value

    def __contains__(self, key : Hashable):
        return key in self.store

    def set(self, key : Hashable, value : Any):
        self.store.__setitem__(key, value)

global_store = ReadOnlyProxy()

class InitialState(ProgramState):
    pass

class ObserveState(ProgramState):
    def __init__(
        self,
        continuation: Callable[[], Callable],
        distribution: Distribution,
        value: Any,
        name: Hashable,
        stack: Tuple[StackFrame]
    ):
        super().__init__(
            continuation=continuation,
            name=name,
            stack=stack
        )
        self.distribution = distribution
        self.value = value

class SampleState(ProgramState):
    def __init__(
        self,
        continuation: Callable[[], Callable],
        distribution: Distribution,
        name: Hashable,
        stack: Tuple[StackFrame]
    ):
        super().__init__(
            continuation=continuation,
            name=name,
            stack=stack
        )
        self.distribution = distribution

class ReturnState(ProgramState):
    def __init__(self, value: Any):
        self.value = value

    def step(self, *args, **kws):
        raise ValueError
