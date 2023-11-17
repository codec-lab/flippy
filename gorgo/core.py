from typing import Any, Callable, Hashable, Tuple, TYPE_CHECKING, TypeVar, Sequence, Union
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

# Thunks and continuations, combined with the CPSInterpreter, form the building
# blocks of the ProgramState abstraction used for defining inference algorithms.

# A thunk is a function that takes no arguments and represents some point in the
# computation that, when executed, will either step to the next point in the
# computation (return a new thunk) or return a ProgramState that can be used to
# modify or inspect the state of the computation.
# By executing a series of thunks using a trampoline, we can run a program.
Thunk = Callable[[], Union['Thunk', 'ProgramState']]

# In continuation-passing style (CPS), a continuation is a parameterizable function that represents
# "what should be done next" after the current function finishes
# and computes a value for the continuation to use
# (i.e., how the program should continue executing).
# Continuations provide a way to represent the stack of a program explicitly.

# In our implementation, continuations either return Thunks to be executed by
# a trampoline, or they return a ProgramState that can be used to modify or inspect
# the computation.
Continuation = Callable[..., Union[Thunk, 'ProgramState']]
VariableName = Hashable
SampleValue = Any
ReturnValue = Any

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
