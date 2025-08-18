from collections.abc import Iterable
from typing import Sequence, Generic, TypeVar, Any, Callable
import math
import random
import abc
from functools import cached_property

from flippy.distributions.support import Support
from flippy.types import Element
from flippy.hashable import hashabledict
from flippy.tools import isclose, ISCLOSE_RTOL, ISCLOSE_ATOL
from flippy.transforms import CPSTransform

class Distribution(Generic[Element]):
    """
    Base class for probability distributions in FlipPy.
    """
    support : Support
    def prob(self, element : Element):
        """Probability of an element in the support of the distribution."""
        return math.exp(self.log_probability(element))

    @abc.abstractmethod
    def sample(self, rng=random, name=None, initial_value=None) -> Element:
        """Sample an element from the distribution."""
        pass

    def observe(self, value: Element) -> None:
        """Observe a value, which is equivalent to conditioning on that value."""
        pass

    def fit(self, *, rng=random, name=None, initial_value: Element = None):
        """Variables returned by this method are fit to maximize marginal likelihood."""
        return initial_value

    @abc.abstractmethod
    def log_probability(self, element : Element) -> float:
        """Log probability of an element"""
        pass

    def expected_value(self, func: Callable[[Element], float] = lambda v : v) -> float:
        """Expected value of the distribution, optionally transformed by `func`."""
        raise NotImplementedError

    def isclose(self, other: "Distribution", *, rtol: float=ISCLOSE_RTOL, atol: float=ISCLOSE_ATOL) -> bool:
        """Check if two distributions are close in terms of their log probabilities."""
        raise NotImplementedError

    def plot(self, ax=None, **kws):
        raise NotImplementedError

    def update(self, data : Sequence[Element]):
        raise NotImplementedError

    def __bool__(self):
        raise ValueError("Cannot convert distribution to bool")

    def total_log_probability(self, data : Sequence[Element]) -> float:
        """Calculate the total log probability of a sequence of data points."""
        return sum(self.log_probability(d) for d in data)
    setattr(total_log_probability, CPSTransform.is_transformed_property, True)

    # This method will be CPS transformed
    def observe_all(self, data : Iterable[Element]):
        """
        Observe all elements in a sequence, equivalent to conditioning on each element.
        """
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
