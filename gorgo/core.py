from typing import Any, Callable, Hashable, Tuple
from gorgo.distributions import Distribution
from gorgo.funcutils import cached_property

# TODO: finalize this interface
class ObservationStatement:
    def __call__(
        self,
        distribution,
        value,
    ):
        pass
observe = ObservationStatement()

############################################
#  Program State
############################################

from collections import namedtuple
StackFrame = namedtuple("StackFrame", "func_src lineno locals")

class ProgramState:
    def __init__(
        self,
        continuation,
        name: Hashable = None,
        stack: Tuple[StackFrame] = None
    ):
        self.continuation = continuation
        self._name = name
        self.stack = stack

    def step(self, *args, **kws):
        thunk = self.continuation(*args, **kws)
        while True:
            next_ = thunk()
            if isinstance(next_, ProgramState):
                return next_
            thunk = next_

    @cached_property
    def name(self):
        if self._name is not None:
            return self._name
        if self.stack is None:
            return None
        return tuple((frame.func_src, frame.lineno) for frame in self.stack)

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
