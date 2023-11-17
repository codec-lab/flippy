from typing import Any, Callable, Dict, Tuple, Union, TYPE_CHECKING, Protocol, Hashable, TypeVar

if TYPE_CHECKING:
    from gorgo.interpreter import StackFrame
    from gorgo.distributions import Distribution
    from gorgo.core import ProgramState
    from gorgo.distributions.random import RandomNumberGenerator, default_rng
    from gorgo.interpreter import CPSInterpreter

Element = TypeVar("Element")

VariableName = Hashable
SampleValue = Any
ReturnValue = Any

Stack = Tuple['StackFrame']

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
NonCPSCallable = Callable[..., Any]

# Type annotation for a sample method of a Distribution
class SampleCallable(Protocol[Element]):
    __self__: 'Distribution[Element]'
    def __call__(self, rng: 'RandomNumberGenerator', name: Hashable) -> 'Element':
        ...

# Type annotation for a observe method of a Distribution
class ObserveCallable(Protocol[Element]):
    __self__: 'Distribution[Element]'
    def __call__(self, value: Element) -> None:
        ...

# Type annotation for a CPS-transformed function
class CPSCallable(Protocol):
    def __call__(
        self,
        *args,
        _cps: 'CPSInterpreter',
        _stack: 'Stack',
        _cont: 'Continuation',
        **kws
    ) -> Union['Thunk', 'ProgramState']:
        ...
