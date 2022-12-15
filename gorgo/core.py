from typing import Sequence, Generic, TypeVar, Any, Callable
import math
import random
import abc
from gorgo.tools import isclose
from gorgo.transforms import CPSTransform


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
    def sample(self, rng=random, _address=None, _cont=None, **kws):
        if _cont is None:
            return self(rng=rng)
        return SampleState(
            continuation=_cont,
            distribution=self
        )
    setattr(sample, CPSTransform.is_transformed_property, True)

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

def observe(distribution : Distribution, value : Element, _address=None, _cont=None, **kws):
    if _cont is None:
        return
    return ObserveState(
        continuation=lambda : _cont(None),
        distribution=distribution,
        value=value
    )
setattr(observe, CPSTransform.is_transformed_property, True)

############################################
#  Program State
############################################

class ProgramState:
    def __init__(
        self,
        continuation,
    ):
        self.continuation = continuation

    def step(self, *args, **kws):
        thunk = self.continuation(*args, **kws)
        while True:
            next_ = thunk()
            if isinstance(next_, ProgramState):
                return next_
            thunk = next_

class InitialState(ProgramState):
    pass

class ObserveState(ProgramState):
    def __init__(self, continuation: Callable[[], Callable], distribution: Distribution, value: Any):
        super().__init__(continuation=continuation)
        self.distribution = distribution
        self.value = value

class SampleState(ProgramState):
    def __init__(self, continuation: Callable[[], Callable], distribution: Distribution):
        super().__init__(continuation=continuation)
        self.distribution = distribution

class ReturnState(ProgramState):
    def __init__(self, value: Any):
        self.value = value

    def step(self, *args, **kws):
        raise ValueError