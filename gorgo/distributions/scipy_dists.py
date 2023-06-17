from typing import Sequence
import random
import numpy as np
import abc
from typing import Union, Tuple, Callable
from functools import cached_property

from gorgo.distributions.base import Distribution, Element
from gorgo.distributions.support import ClosedInterval
from gorgo.distributions.random import RandomNumberGenerator, default_rng

from scipy.stats import rv_continuous, rv_discrete
from scipy.stats import norm, uniform, beta, gamma, poisson

__all__ = [
    "Normal",
    "Uniform",
    "Gamma",
    "Beta",
]

class ScipyContinuousDistribution(Distribution):
    loc = 0
    scale = 1
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
            def __init__(self, *args, loc=0, scale=1):
                self.loc = loc
                self.scale = scale
                self.args = args

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
        shape : Union[int, Tuple] = 1,
        *,
        rng : RandomNumberGenerator = default_rng,
        name=None
    ) -> Sequence[Element]:
        sample = self.base_distribution.rvs(
            *self.args,
            loc=self.loc,
            scale=self.scale,
            size=shape,
            random_state=rng.np
        )
        if shape == 1:
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

# Common parameterizations of scipy distributions

class Normal(ScipyContinuousDistribution):
    base_distribution = norm
    def __init__(self, mean=0, sd=1):
        self.loc = self.mean = mean
        self.scale = self.sd = sd
    def __repr__(self) -> str:
        return f"Normal(mean={self.loc}, sd={self.scale})"

class Uniform(ScipyContinuousDistribution):
    base_distribution = uniform
    def __init__(self, low=0, high=1):
        self.loc = self.low = low
        self.scale = self.high = high
    def __repr__(self) -> str:
        return f"Uniform(low={self.loc}, high={self.scale})"

class Gamma(ScipyContinuousDistribution):
    base_distribution = gamma
    def __init__(self, shape=1, rate=1):
        self.scale = 1/rate
        self.args = (shape,)
        self.shape = shape
        self.rate = rate
    def __repr__(self) -> str:
        return f"Gamma(shape={self.shape}, rate={self.rate})"

class Beta(ScipyContinuousDistribution):
    base_distribution = beta
    def __init__(self, alpha=1, beta=1):
        self.args = (alpha, beta)
        self.alpha = alpha
        self.beta = beta
    def __repr__(self) -> str:
        return f"Beta(alpha={self.alpha}, beta={self.beta})"
