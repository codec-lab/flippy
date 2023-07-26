from typing import Sequence
import random
import numpy as np
import abc
from typing import Union, Tuple, Callable
from functools import cached_property

from gorgo.distributions.base import Distribution, Element, FiniteDistribution, Multivariate
from gorgo.distributions.support import ClosedInterval, CrossProduct
from gorgo.distributions.random import RandomNumberGenerator, default_rng
from gorgo.tools import isclose

from scipy.stats import rv_continuous, rv_discrete
from scipy.stats import norm, uniform, beta, gamma, poisson, bernoulli

__all__ = [
    "Normal",
    "Uniform",
    "Gamma",
    "Beta",
    "Bernoulli"
]

class ScipyContinuousDistribution(Distribution, Multivariate):
    loc = 0
    scale = 1
    size = 1
    args = ()

    @property
    @abc.abstractmethod
    def base_distribution(self) -> rv_continuous:
        pass

    @classmethod
    def create_distribution_class(
        cls,
        base_distribution : rv_continuous,
        name : str,
    ):
        class NewDistribution(cls):
            def __init__(self, *args, loc=0, scale=1, size=1):
                self.loc = loc
                self.scale = scale
                self.args = args
                self.size = size

            @property
            def base_distribution(self):
                return base_distribution

            def __repr__(self) -> str:
                args = ", ".join(list(map(str, self.args)) + [f"loc={self.loc}", f"scale={self.scale}"])
                return f"{self.__class__.__name__}({args})"

        NewDistribution.__name__ = name
        return NewDistribution

    def sample(
        self,
        rng : RandomNumberGenerator = default_rng,
        name=None
    ) -> Sequence[Element]:
        sample = self.base_distribution.rvs(
            *self.args,
            loc=self.loc,
            scale=self.scale,
            size=self.size,
            random_state=rng.np
        )
        if self.size == 1:
            return sample[0]
        return sample

    def observe(self, value : Sequence[Element]) -> None:
        pass

    def log_probabilities(self, element : Sequence[Element]) -> Sequence[float]:
        return self.base_distribution.logpdf(
            element,
            *self.args,
            loc=self.loc,
            scale=self.scale,
        )

    def log_probability(self, element : Sequence[Element]) -> float:
        logprobs = self.log_probabilities(element)
        if isinstance(logprobs, float):
            return logprobs
        return sum(logprobs)

    def expected_value(self, func: Callable[[Element], float] = lambda v : v) -> float:
        return self.base_distribution.expect(
            func=func,
            args=self.args,
            loc=self.loc,
            scale=self.scale,
        )

    def isclose(self, other: "Distribution") -> bool:
        if not isinstance(other, ScipyContinuousDistribution):
            return False
        return self.base_distribution.__class__ == other.base_distribution.__class__ and \
            self.args == other.args

    @cached_property
    def support(self) -> ClosedInterval:
        return ClosedInterval(
            *self.base_distribution.support(*self.args, loc=self.loc, scale=self.scale)
        )

    def plot(self, ax=None, bins=100, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np
        if ax is None:
            fig, ax = plt.subplots()
        x = np.linspace(0, 1, 1000)
        ax.plot(x, [self.prob(i) for i in x], **kwargs)
        return ax

# Common parameterizations of scipy distributions

class Normal(ScipyContinuousDistribution):
    base_distribution = norm
    def __init__(self, mean=0, sd=1, size=1):
        self.loc = self.mean = mean
        self.scale = self.sd = sd
        self.size = size
    def __repr__(self) -> str:
        return f"Normal(mean={self.loc}, sd={self.scale}, size={self.size}))"

class Uniform(ScipyContinuousDistribution):
    base_distribution = uniform
    def __init__(self, low=0, high=1, size=1):
        self.loc = self.low = low
        self.high = high
        self.scale = high - low
        self.size = size
    def __repr__(self) -> str:
        return f"Uniform(low={self.low}, high={self.high}, size={self.size})"

class Gamma(ScipyContinuousDistribution):
    base_distribution = gamma
    def __init__(self, shape=1, rate=1, size=1):
        self.scale = 1/rate
        self.args = (shape,)
        self.shape = shape
        self.rate = rate
        self.size = size
    def __repr__(self) -> str:
        return f"Gamma(shape={self.shape}, rate={self.rate}, size={self.size})"

class Beta(ScipyContinuousDistribution):
    base_distribution = beta
    def __init__(self, alpha=1, beta=1, size=1):
        assert not isclose(alpha, 0) and not isclose(beta, 0), "alpha and beta must be non-zero"
        self.args = (alpha, beta)
        self.alpha = alpha
        self.beta = beta
        self.size = size

    def __repr__(self) -> str:
        return f"Beta(alpha={self.alpha}, beta={self.beta}, size={self.size})"


class Bernoulli(FiniteDistribution, Multivariate):
    def __init__(self, p=0.5, size=1):
        self.p = p
        self.size = size

    @cached_property
    def support(self) -> Tuple:
        if self.size == 1:
            return (0, 1)
        return CrossProduct([(0, 1)] * self.size)

    def __repr__(self) -> str:
        return f"Bernoulli(p={self.p}, size={self.size})"

    def sample(self, rng : RandomNumberGenerator = default_rng, name=None) -> Sequence[Element]:
        s = bernoulli.rvs(self.p, size=self.size, random_state=rng.np)
        if self.size == 1:
            return s[0]
        return tuple(s)

    def observe(self, value : Sequence[Element]) -> None:
        pass

    def log_probabilities(self, element : Sequence[Element]) -> Sequence[float]:
        return bernoulli.logpmf(element, self.p)

    def log_probability(self, element : Sequence[Element]) -> float:
        logprobs = self.log_probabilities(element)
        if isinstance(logprobs, float):
            return logprobs
        return sum(logprobs)

    def expected_value(self, func: Callable[[Element], float] = lambda v : v) -> float:
        return self.p

    def isclose(self, other: "Distribution") -> bool:
        if not isinstance(other, Bernoulli):
            return False
        return self.p == other.p
