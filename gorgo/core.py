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
        self.global_store = None

    def step(self, *args, **kws):
        thunk = self.continuation(*args, **kws)
        global_store = {**self.global_store} if self.global_store is not None else {}
        while True:
            next_ = thunk()
            if isinstance(next_, ProgramState):
                next_.global_store = global_store
                return next_
            elif isinstance(next_, GlobalStoreIncludes):
                next_ = next_.continuation(next_.key in global_store)
            elif isinstance(next_, GlobalStoreRead):
                next_ = next_.continuation(global_store.get(next_.key))
            elif isinstance(next_, GlobalStoreWrite):
                global_store[next_.key] = next_.value
                next_ = (lambda last_next_ : (lambda : last_next_.continuation(None)))(next_)
            thunk = next_

    @cached_property
    def name(self):
        if self._name is not None:
            return self._name
        if self.stack is None:
            return None
        return tuple((frame.func_src, frame.lineno) for frame in self.stack)

class GlobalStoreRead(dict):
    def __init__(self, continuation : Continuation, key : Hashable):
        self.continuation = continuation
        self.key = key

class GlobalStoreWrite(dict):
    def __init__(self, continuation : Continuation, key : Hashable, value : Any):
        self.continuation = continuation
        self.key = key
        self.value = value

class GlobalStoreIncludes:
    def __init__(self, continuation : Continuation, key : Hashable):
        self.continuation = continuation
        self.key = key

class GlobalStore(dict):
    def read(self, key : Hashable):
        return self[key]

    def write(self, key : Hashable, value : Any):
        self[key] = value

    def includes(self, key : Hashable):
        return key in self

global_store = GlobalStore()

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
