from typing import Callable, TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from gorgo.core import ProgramState
from gorgo.types import CPSCallable, Continuation, Stack
from gorgo.transforms import CPSTransform
from gorgo.hashable import hashabledict
if TYPE_CHECKING:
    from gorgo.interpreter import CPSInterpreter

class EnterCallState(ProgramState):
    def __init__(
        self,
        f: CPSCallable,
        args: Tuple,
        kwargs: Dict,
        continuation: Continuation=None,
        cps: 'CPSInterpreter'=None,
        stack: Stack=None,
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

class ExitCallState(ProgramState):
    def __init__(
        self,
        f: CPSCallable,
        args: Tuple,
        kwargs: Dict,
        value: Any,
        continuation: Continuation=None,
        cps: 'CPSInterpreter'=None,
        stack: Stack=None,
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

def call_event(
    f: CPSCallable,
    args: Tuple,
    kwargs: Dict,
    _cont: Continuation=None,
    _cps: 'CPSInterpreter'=None,
    _stack: Stack=None,
):
    return EnterCallState(
        f=f,
        args=args,
        kwargs=kwargs,
        continuation=lambda : _cont(None),
        cps=_cps,
        stack=_stack,
    )
setattr(call_event, CPSTransform.is_transformed_property, True)

def exit_event(
    f: CPSCallable,
    args: Tuple,
    kwargs: Dict,
    value: Any,
    _cont: Continuation=None,
    _cps: 'CPSInterpreter'=None,
    _stack: Stack=None,
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
setattr(exit_event, CPSTransform.is_transformed_property, True)

def register_call_entryexit(f: CPSCallable) -> CPSCallable:
    def call_entryexit_wrapper(*args, **kwargs):
        kwargs = hashabledict(kwargs)
        call_event(f, args, kwargs)
        res = f(*args, **kwargs)
        exit_event(f, args, kwargs, res)
        return res
    return call_entryexit_wrapper
