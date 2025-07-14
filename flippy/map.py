from typing import Sequence, Tuple, Callable, TypeVar, List, TYPE_CHECKING

from flippy.core import ProgramState
from flippy.types import Continuation, CPSCallable
from flippy.interpreter import CPSInterpreter
from flippy.transforms import CPSTransform

from flippy.callentryexit import register_call_entryexit
if TYPE_CHECKING:
    from flippy.interpreter import Stack

T = TypeVar('T')
I = TypeVar('I')

def recursive_map(fn: Callable[[I],T], iter: Sequence[I]) -> List[T]:
    if not iter:
        return []
    return [fn(iter[0])] + recursive_map(fn, iter[1:])

def independent_map(func: Callable[[I],T], iterator: Sequence[I]) -> Tuple[T, ...]:
    """
    This gives the inference algorithm access to a mapped function
    applied to each iterate using the EnterCallState/ExitCallState interface.
    We do this by:

    1. Sending a `MapEnter` signal to the algorithm to see if it wants access to
    the mapped function applied to each iterate. If not, it just uses `recursive_map`.

    2. If it does, we intercept and then reinstate the CPS transform to construct a
    sequence of EnterCallState/ExitCallState blocks that
    *does not save non-global state between each iterate*. The inference algorithm
    is responsible for handing return values from `ExitCallState`s.

    3. Sending a `MapExit` signal to the algorithm that accepts the result of applying
    the mapped function over each iterate.
    """
    send_call_entryexit = map_enter_event()
    if send_call_entryexit:
        return _independent_map(func, iterator)
    else:
        return tuple(recursive_map(func, iterator))

class MapEnter(ProgramState):
    pass

class MapExit(ProgramState):
    pass

def map_enter_event(*, _stack=None, _cps=None, _cont=None):
    return MapEnter(
        continuation=lambda send_call_entryexit=False : _cont(send_call_entryexit),
        stack=_stack,
        cps=_cps,
    )
setattr(map_enter_event, CPSTransform.is_transformed_property, True)

def _independent_map(
    func: CPSCallable,
    iterator: Sequence,
    *,
    _stack: 'Stack' = None,
    _cps: CPSInterpreter = None,
    _cont: Continuation = None
):
    def construct_cont(next_cont, i):
        # note we don't actually use the return value of each iterate
        # here (_res). The inference algorithm is responsible for handling
        # that when it receives an `ExitCallState` message.
        return lambda : _cps.interpret(
            _independent_map_iter,
            cont=lambda _res : next_cont(),
            stack=_stack,
            func_src="INDEPENDENT_MAP_ITER",
            locals_={},
            lineno=0
        )(func, i)

    # we start from the end and then iterate backwards
    next_cont = lambda : MapExit(
        continuation=lambda map_result: _cont(map_result),
        stack=_stack,
        cps=_cps,
    )
    for i in iterator[::-1]:
        last_cont = construct_cont(next_cont, i)
        next_cont = last_cont
    return last_cont
setattr(_independent_map, CPSTransform.is_transformed_property, True)

# Here's where each individual function call lives. It is wrapped in an entry/exit
# wrapper and gets CPS transformed.
def _independent_map_iter(func, i):
    return func(i)
_independent_map_iter = CPSInterpreter().non_cps_callable_to_cps_callable(_independent_map_iter)
_independent_map_iter = register_call_entryexit(_independent_map_iter)
