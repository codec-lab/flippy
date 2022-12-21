from typing import Sequence, Generic, TypeVar, Any, Callable, Hashable, Tuple
import math
import random
import abc
from gorgo.tools import isclose
from gorgo.funcutils import cached_property

############################################
#  Sampling and observations
############################################

Element = TypeVar("Element")

class Distribution(Generic[Element]):
    support: Sequence[Element]
    def log_probability(self, element : Element) -> float:
        pass
    def sample(self, name=None) -> Element:
        pass
    def isclose(self, other: "Distribution") -> bool:
        full_support = set(self.support) | set(other.support)
        return all(
            isclose(self.log_probability(s), other.log_probability(s))
            for s in full_support
        )

class StochasticPrimitive(Distribution):
    support : Sequence
    @abc.abstractmethod
    def __call__(self, *params, rng=random):
        pass
    def sample(self, rng=random, name=None):
        return self(rng=rng)

class Bernoulli(StochasticPrimitive):
    support = (True, False)
    def __init__(self, p=.5):
        self.p = p
    def __call__(self, rng=random):
        if rng.random() <= self.p:
            return True
        return False
    def log_probability(self, element):
        return {
            True: math.log(self.p),
            False: math.log(1 - self.p) if self.p != 1.0 else float('-inf')
        }[element]

class Multinomial(StochasticPrimitive):
    def __init__(self, support, *, probabilities=None, weights=None):
        if probabilities is not None:
            assert isclose(sum(probabilities), 1)
        elif weights is not None:
            probabilities = [w / sum(weights) for w in weights]
            del weights
        else:
            probabilities = [1/len(support) for _ in support]
        self.support = support
        self._probabilities = probabilities

    def __call__(self, rng=random):
        return rng.choices(self.support, weights=self._probabilities, k=1)[0]

    def log_probability(self, element):
        try:
            return math.log(self._probabilities[self.support.index(element)])
        except ValueError:
            return float('-inf')

    def __repr__(self):
        return repr({
            s: self.log_probability(s)
            for s in self.support
        })

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