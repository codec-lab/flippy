from typing import Any, Callable, Hashable, Tuple, TYPE_CHECKING, TypeVar, Sequence, Union
from flippy.distributions import Distribution
from flippy.funcutils import cached_property
from flippy.hashable import hashabledict, hashablelist, hashableset

from flippy.types import Continuation, VariableName, ReturnValue

if TYPE_CHECKING:
    from flippy.interpreter import CPSInterpreter, Stack

############################################
#  Program State
############################################

class ProgramState:
    """
    A program state represents a point in the execution of a program
    (e.g., the current call stack, memory, and line number) that is
    exposed to an external interpreter (e.g., an inference algorithm).
    An external interpreter can control the behavior of a program using the
    `step` method. An interpreter can also access readable attributes and
    meta-data associated with program state.
    Internally, a program state stores a continuation that is used to resume
    execution when the interpreter calls `step`.
    """
    value = None
    def __init__(
        self,
        continuation : 'Continuation' = None,
        name: 'VariableName' = None,
        stack: 'Stack' = None,
        cps : 'CPSInterpreter' = None,
        init_global_store : 'GlobalStore' = None,
    ):
        self.continuation = continuation
        self._name = name
        self.stack = stack
        self.init_global_store = init_global_store
        self.cps = cps

    def set_init_global_store(self, global_store : 'GlobalStore', force=False):
        assert self.init_global_store is None or force, "Cannot set global store twice"
        self.init_global_store = global_store

    def step(self, *args, **kws) -> 'ProgramState':
        """
        Uses a trampoline to execute a sequence of thunks until
        a ProgramState is encountered.
        """
        next_ = self.continuation(*args, **kws)
        global_store = self.init_global_store.copy()
        with self.cps.set_global_store(global_store):
            while True:
                if callable(next_):
                    next_ = next_()
                elif isinstance(next_, ProgramState):
                    next_.set_init_global_store(global_store)
                    return next_
                else:
                    raise TypeError(f"Unknown type {type(next_)}")

    @cached_property
    def name(self) -> 'VariableName':
        if self._name is not None:
            return self._name
        if self.stack is None:
            return None
        return self.stack.without_locals()

    def __eq__(self, other: 'ProgramState'):
        if not isinstance(other, ProgramState):
            return False
        if self._hash != other._hash:
            return False
        return (
            self.__class__ == other.__class__ and
            self.stack == other.stack and
            self.init_global_store.store == other.init_global_store.store
        )

    @cached_property
    def _hash(self):
        return hash((self.__class__, self.stack, hashabledict(self.init_global_store.store), self.value))

    def __hash__(self):
        return self._hash

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
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__ = state

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
    def __init__(
        self,
        continuation : 'Continuation' = None,
        cps : 'CPSInterpreter' = None,
    ):
        super().__init__(
            continuation=continuation,
            cps=cps,
            init_global_store=GlobalStore()
        )

class ObserveState(ProgramState):
    def __init__(
        self,
        continuation: 'Continuation',
        distribution: Distribution,
        value: Any,
        name: 'VariableName',
        stack: 'Stack',
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
        continuation: 'Continuation',
        distribution: Distribution,
        name: 'VariableName',
        stack: 'Stack',
        cps : 'CPSInterpreter',
        initial_value: Any,
        fit: bool
    ):
        super().__init__(
            continuation=continuation,
            name=name,
            stack=stack,
            cps=cps
        )
        self.distribution = distribution
        self.initial_value = initial_value
        self.fit = fit

class ReturnState(ProgramState):
    def __init__(self, value: 'ReturnValue', stack: 'Stack'):
        super().__init__(stack=stack, name="RETURN_STATE")
        if isinstance(value, dict):
            value = hashabledict(value)
        elif isinstance(value, list):
            value = hashablelist(value)
        elif isinstance(value, set):
            value = hashableset(value)
        self.value = value

    def step(self, *args, **kws):
        raise ValueError
