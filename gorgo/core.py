from typing import Any, Callable, Hashable, Tuple, TYPE_CHECKING, TypeVar
from gorgo.distributions import Distribution
from gorgo.funcutils import cached_property
from gorgo.hashable import hashabledict, hashablelist, hashableset

if TYPE_CHECKING:
    from gorgo.interpreter import CPSInterpreter

############################################
#  Program State
############################################

from collections import namedtuple
StackFrame = namedtuple("StackFrame", "func_src lineno locals")

# a continuation is a function that takes a value and returns a thunk
Continuation = Callable[..., Callable[[], Any]]
VariableName = Hashable
SampleValue = Any
ReturnValue = Any

class ProgramState:
    def __init__(
        self,
        continuation : Continuation = None,
        name: VariableName = None,
        stack: Tuple[StackFrame] = None,
        cps : 'CPSInterpreter' = None
    ):
        self.continuation = continuation
        self._name = name
        self.stack = stack
        self.init_global_store = GlobalStore()
        self.cps = cps

    def step(self, *args, **kws) -> 'ProgramState':
        next_ = self.continuation(*args, **kws)
        global_store = self.init_global_store.copy()
        with self.cps.set_global_store(global_store):
            while True:
                if callable(next_):
                    next_ = next_()
                elif isinstance(next_, ProgramState):
                    next_.init_global_store = global_store
                    return next_
                else:
                    raise TypeError(f"Unknown type {type(next_)}")

    @cached_property
    def name(self) -> VariableName:
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
        name: VariableName,
        stack: Tuple[StackFrame],
        cps : 'CPSInterpreter'
    ):
        super().__init__(
            continuation=continuation,
            name=name,
            stack=stack,
            cps=cps
        )
        self.distribution = distribution
        self.value = value

class SampleState(ProgramState):
    def __init__(
        self,
        continuation: Callable[[], Callable],
        distribution: Distribution,
        name: VariableName,
        stack: Tuple[StackFrame],
        cps : 'CPSInterpreter'
    ):
        super().__init__(
            continuation=continuation,
            name=name,
            stack=stack,
            cps=cps
        )
        self.distribution = distribution

class ReturnState(ProgramState):
    def __init__(self, value: ReturnValue):
        if isinstance(value, dict):
            value = hashabledict(value)
        elif isinstance(value, list):
            value = hashablelist(value)
        elif isinstance(value, set):
            value = hashableset(value)
        self.value = value
        self._name = "RETURN_STATE"

    def step(self, *args, **kws):
        raise ValueError
