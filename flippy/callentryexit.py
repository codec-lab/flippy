from typing import Callable, TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from flippy.core import ProgramState
from flippy.types import CPSCallable, Continuation, Thunk
from flippy.transforms import CPSTransform, CPSFunction
from flippy.hashable import hashabledict
if TYPE_CHECKING:
    from flippy.interpreter import CPSInterpreter, Stack

class EnterCallState(ProgramState):
    def __init__(
        self,
        f: CPSFunction,
        args: Tuple,
        kwargs: Dict,
        continuation: Callable[[bool, Any], Union[Thunk, ProgramState]]=None,
        cps: 'CPSInterpreter'=None,
        stack: 'Stack'=None,
    ):
        super().__init__(
            continuation=continuation,
            name=(f, args, hashabledict(kwargs)),
            cps=cps,
            stack=stack,
        )
        self.function = f
        self.args = args
        self.kwargs = kwargs

    @property
    def is_root_call(self) -> bool:
        return len(self.stack) == 1

    def skip(self, value) -> 'ExitCallState':
        return self.step(False, value)

    def step(self, run_func: bool = True, res: Any = None) -> 'ProgramState':
        return ProgramState.step(self, run_func, res)

class ExitCallState(ProgramState):
    def __init__(
        self,
        f: CPSFunction,
        args: Tuple,
        kwargs: Dict,
        value: Any,
        continuation: Continuation=None,
        cps: 'CPSInterpreter'=None,
        stack: 'Stack'=None,
    ):
        super().__init__(
            continuation=continuation,
            name=(f, args, hashabledict(kwargs), value),
            cps=cps,
            stack=stack,
        )
        self.function = f
        self.args = args
        self.kwargs = kwargs
        self.value = value

def enter_call_event(
    f: CPSFunction,
    args: Tuple,
    kwargs: Dict,
    _cont: Continuation=None,
    _cps: 'CPSInterpreter'=None,
    _stack: 'Stack'=None,
):
    return EnterCallState(
        f=f,
        args=args,
        kwargs=kwargs,
        continuation=lambda run_func=True, res=None: _cont((run_func, res)),
        cps=_cps,
        stack=_stack,
    )
setattr(enter_call_event, CPSTransform.is_transformed_property, True)

def exit_call_event(
    f: CPSFunction,
    args: Tuple,
    kwargs: Dict,
    value: Any,
    _cont: Continuation=None,
    _cps: 'CPSInterpreter'=None,
    _stack: 'Stack'=None,
):
    return ExitCallState(
        f=f,
        args=args,
        kwargs=kwargs,
        value=value,
        continuation=lambda : _cont(None),
        cps=_cps,
        stack=_stack,
    )
setattr(exit_call_event, CPSTransform.is_transformed_property, True)

def register_call_entryexit(f: CPSCallable) -> CPSCallable:
    assert isinstance(f, CPSFunction), \
        f'Error registering {f.__name__}: `register_call_entryexit` can only be applied to transformed functions'
    def call_entryexit_wrapper(*args, **kwargs):
        kwargs = hashabledict(kwargs)
        run_func, res = enter_call_event(f, args, kwargs)
        if run_func:
            res = f(*args, **kwargs)
        exit_call_event(f, args, kwargs, res)
        return res
    return call_entryexit_wrapper
