from collections.abc import Iterable
from typing import Sequence, Generic, TypeVar, Any, Callable
import math
import random
import abc
from flippy.tools import isclose, ISCLOSE_RTOL, ISCLOSE_ATOL
from functools import cached_property

from flippy.distributions.support import Support
from flippy.types import Element
from flippy.hashable import hashabledict

class Distribution(Generic[Element]):
    support : Support
    def prob(self, element : Element):
        return math.exp(self.log_probability(element))

    @abc.abstractmethod
    def sample(self, rng=random, name=None, initial_value=None) -> Element:
        pass

    def observe(self, value) -> None:
        pass

    def fit(self, *, rng=random, name=None, initial_value: Element = None):
        return initial_value

    @abc.abstractmethod
    def log_probability(self, element : Element) -> float:
        pass

    def expected_value(self, func: Callable[[Element], float] = lambda v : v) -> float:
        raise NotImplementedError

    def isclose(self, other: "Distribution", *, rtol: float=ISCLOSE_RTOL, atol: float=ISCLOSE_ATOL) -> bool:
        raise NotImplementedError

    def plot(self, ax=None, **kws):
        raise NotImplementedError

    def update(self, data : Sequence[Element]):
        raise NotImplementedError

    def __bool__(self):
        raise ValueError("Cannot convert distribution to bool")

    def total_log_probability(self, data : Sequence[Element]) -> float:
        return sum(self.log_probability(d) for d in data)

    # This method will be CPS transformed
    def observe_all(self, data : Iterable[Element]):
        _factor_dist.observe(self.total_log_probability(data))


class FactorDistribution(Distribution):
    def __init__(self):
        pass

    def sample(self, rng, name, initial_value=None):
        return 0

    def log_probability(self, element : float) -> float:
        #workaround for arbitrary scores
        return element

_factor_dist = FactorDistribution()


class FiniteDistribution(Distribution[Element]):
    support: Sequence[Element]

    @cached_property
    def probabilities(self):
        return tuple(self.prob(e) for e in self.support)

    def isclose(self, other: "FiniteDistribution", *, rtol: float=ISCLOSE_RTOL, atol: float=ISCLOSE_ATOL) -> bool:
        full_support = set(self.support) | set(other.support)
        return all(
            isclose(self.log_probability(s), other.log_probability(s), rtol=rtol, atol=atol)
            for s in full_support
        )

    def items(self):
        yield from zip(self.support, self.probabilities)

    def expected_value(self, func: Callable[[Element], Any] = lambda v : v) -> Any:
        return sum(
            p*func(s)
            for s, p in self.items()
        )

    def __getitem__(self, element):
        return self.prob(element)

    def as_dict(self):
        return dict(zip(self.support, self.probabilities))

    def __len__(self):
        return len(self.support)

    def keys(self):
        yield from self.support

    def values(self):
        yield from self.probabilities

    def items(self):
        yield from zip(self.support, self.probabilities)

    def __iter__(self):
        yield from self.support

    def __hash__(self):
        return hash(hashabledict(self.as_dict()))

    def __eq__(self, other):
        if not isinstance(other, FiniteDistribution):
            return False
        return self.as_dict() == other.as_dict()

class Multivariate:
    size : int = 1
