from typing import Sequence, Protocol, Collection, TypeVar
from gorgo.types import Continuation, Thunk, Stack, CPSCallable
from gorgo.core import ProgramState
from gorgo.interpreter import CPSInterpreter
from gorgo.transforms import CPSTransform

from gorgo.hashable import hashabledict, hashablelist, hashableset

IterInput = TypeVar("IterInput")
IterOutput = TypeVar("IterOutput")

def MapIterStartContinuation(Protocol):
    def __call__(self, i : IterInput) -> Thunk:
        ...

def MapFinishContinuation(Protocol):
    def __call__(self, values : Sequence[IterOutput]) -> Thunk:
        ...

class NoGlobalStoreProgramState(ProgramState):
    def step(self, *args, **kws) -> 'ProgramState':
        """
        Uses a trampoline to execute a sequence of thunks until
        a ProgramState is encountered.
        """
        next_ = self.continuation(*args, **kws)
        # TODO: throw more informative error if global store is accessed
        # TODO: refactor base programstate to make global store writing optional
        # or make a way to access global store
        while True:
            if callable(next_):
                next_ = next_()
            elif isinstance(next_, ProgramState):
                return next_
            else:
                raise TypeError(f"Unknown type {type(next_)}")

class MapIterStart(NoGlobalStoreProgramState):
    def __init__(
        self,
        iter_continuation : 'MapIterStartContinuation',
        iterator : Sequence[IterInput],
        map_finish_program_state : 'MapFinish' = None,
    ):
        self.continuation = iter_continuation
        self.iterator = iterator
        self.map_finish_program_state = map_finish_program_state

class MapIterEnd(NoGlobalStoreProgramState):
    def __init__(
        self,
        value : IterOutput,
    ):
        if isinstance(value, dict):
            value = hashabledict(value)
        elif isinstance(value, list):
            value = hashablelist(value)
        elif isinstance(value, set):
            value = hashableset(value)
        self.value = value

class MapFinish(NoGlobalStoreProgramState):
    def __init__(
        self,
        finish_continuation : 'MapFinishContinuation',
    ):
        self.continuation = finish_continuation

def independent_map(
    func: CPSCallable,
    iterator: Sequence,
    *,
    _stack: Stack = None,
    _cps: CPSInterpreter = None,
    _cont: Continuation = None
):
    map_finish_program_state = MapFinish(
        finish_continuation=_cont,
    )
    def iter_end_continuation(result: IterOutput):
        return MapIterEnd(
            value=result
        )
    def iter_continuation(i: IterInput):
        return lambda : _cps.interpret(func, cont=iter_end_continuation)(i)
    return MapIterStart(
        iter_continuation=iter_continuation,
        iterator=iterator,
        map_finish_program_state=map_finish_program_state,
    )
setattr(independent_map, CPSTransform.is_transformed_property, True)
