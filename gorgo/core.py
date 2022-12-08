from typing import Sequence, Generic, TypeVar, Any
import math
import random
import abc
import dataclasses
from gorgo.tools import isclose

############################################
#  Sampling and observations
############################################

Element = TypeVar("Element")

class Distribution(Generic[Element]):
    support: Sequence[Element]
    def log_probability(self, element : Element) -> float:
        pass
    def sample(self) -> Element:
        pass

class StochasticPrimitive(Distribution):
    support : Sequence
    @abc.abstractmethod
    def __call__(self, *params, rng=random):
        pass
    def sample(self, rng=random):
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
#  Messages and Program State
############################################

@dataclasses.dataclass
class ProgramMessage:
    address : tuple

@dataclasses.dataclass
class StartingMessage(ProgramMessage):
    pass

@dataclasses.dataclass
class ObserveMessage(ProgramMessage):
    distribution: Distribution
    value: Any

@dataclasses.dataclass
class SampleMessage(ProgramMessage):
    distribution: Distribution

@dataclasses.dataclass
class ReturnMessage(ProgramMessage):
    value : Any

class ProgramState:
    def __init__(
        self,
        continuation,
        message,
        is_returned=False
    ):
        self.continuation = continuation
        self.message = message
        self._is_returned = is_returned

    def is_returned(self):
        return self._is_returned

    def step(self, *args, **kws):
        if self.is_returned():
            raise ValueError
        thunk = self.continuation(*args, **kws)
        while True:
            next_ = thunk()
            if isinstance(next_, ProgramState):
                return next_
            thunk = next_
