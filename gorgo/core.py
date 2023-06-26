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
        next_ = self.continuation(*args, **kws)
        global_store = GlobalStore({**self.global_store} if self.global_store is not None else {})
        # HACK: Is there no other reasonable way to access interpreter?
        _cps = next_.__globals__['_cps']
        _cps.global_store_proxy.proxied = global_store
        while True:
            if callable(next_):
                next_ = next_()
            elif isinstance(next_, ProgramState):
                next_.global_store = global_store
                return next_
            # elif isinstance(next_, GlobalStoreEvent):
            #     if isinstance(next_, GlobalStoreIncludes):
            #         next_ = next_.continuation(next_.key in global_store)
            #     elif isinstance(next_, GlobalStoreRead):
            #         next_ = next_.continuation(global_store.get(next_.key))
            #     elif isinstance(next_, GlobalStoreWrite):
            #         global_store[next_.key] = next_.value
            #         next_ = (lambda last_next_ : (lambda : last_next_.continuation(None)))(next_)
            else:
                raise TypeError(f"Unknown type {type(next_)}")
            # thunk = next_

    @cached_property
    def name(self):
        if self._name is not None:
            return self._name
        if self.stack is None:
            return None
        return tuple((frame.func_src, frame.lineno) for frame in self.stack)

class GlobalStoreEvent:
    pass

class GlobalStoreRead(GlobalStoreEvent):
    def __init__(self, continuation : Continuation, key : Hashable):
        self.continuation = continuation
        self.key = key

class GlobalStoreWrite(GlobalStoreEvent):
    def __init__(self, continuation : Continuation, key : Hashable, value : Any):
        self.continuation = continuation
        self.key = key
        self.value = value

class GlobalStoreIncludes(GlobalStoreEvent):
    def __init__(self, continuation : Continuation, key : Hashable):
        self.continuation = continuation
        self.key = key

class ReadOnlyProxy(object):
    def __init__(self):
        self.proxied = None
    def __getattr__(self, name):
        if self.proxied is None:
            raise NotImplementedError("Proxying to None")
        return getattr(self.proxied, name)

class GlobalStore(dict):
    def read(self, key : Hashable):
        return self[key]

    def write(self, key : Hashable, value : Any):
        self[key] = value

    def includes(self, key : Hashable):
        return key in self

global_store = ReadOnlyProxy()

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
