from typing import Sequence
import random
import numpy as np
import sympy as sp
import abc
from typing import Union, Tuple, Callable
from functools import cached_property

from flippy.distributions.base import Distribution, Element, FiniteDistribution, Multivariate
from flippy.distributions.support import ClosedInterval, CrossProduct
from flippy.distributions.random import RandomNumberGenerator, default_rng
from flippy.tools import isclose

from scipy.stats import rv_continuous, rv_discrete
from scipy.stats import norm, uniform, beta, gamma, poisson, bernoulli, multivariate_normal, \
    invwishart

__all__ = [
    "Normal",
    "Uniform",
    "Gamma",
    "Beta",
    "Bernoulli",
    "NormalNormal",
    "MultivariateNormalNormal",
    "MultivariateNormal",
    "InverseWishart",
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
        name=None,
        initial_value=None
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

    def plot(self, ax=None, xlim=(None, None), **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np
        if ax is None:
            fig, ax = plt.subplots()
        if xlim[0] is None:
            xlim = (max(self.support.start, -10), xlim[1])
        if xlim[1] is None:
            xlim = (xlim[0], min(self.support.end, 10))
        x = np.linspace(*xlim, 1000)
        ax.plot(x, [self.prob(i) for i in x], **kwargs)
        return ax

    def __hash__(self):
        return hash((self.__class__, self.args, self.loc, self.scale))

    def __eq__(self, other: "ScipyContinuousDistribution"):
        return self.__class__ == other.__class__ and \
            self.args == other.args and \
            self.loc == other.loc and \
            self.scale == other.scale

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
        self.p = float(p)
        self.size = size

    @cached_property
    def support(self) -> Tuple:
        if self.size == 1:
            return (0, 1)
        return CrossProduct([(0, 1)] * self.size)

    def __repr__(self) -> str:
        return f"Bernoulli(p={self.p}, size={self.size})"

    def sample(
        self,
        rng : RandomNumberGenerator = default_rng,
        name=None,
        initial_value=None
    ) -> Sequence[Element]:
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

class NormalNormal(Multivariate):
    def __init__(self, *, prior_mean=0, prior_sd=1, sd=1, size=1):
        self.prior_mean = np.array(prior_mean)
        self.prior_sd = np.array(prior_sd)
        self.sd = np.array(sd)
        assert isinstance(size, int)
        assert np.shape(self.prior_mean) == np.shape(self.prior_sd) == np.shape(self.sd) == ()
        self.size = size

    def __repr__(self) -> str:
        return f"NormalNormal(prior_mean={self.prior_mean}, prior_sd={self.prior_sd}, sd={self.sd}, size={self.size})"

    #first sample mu aka mean and then sample y aka x from estimated mu
    #check scipy norm documentation for why loc / general things
    def sample(
        self,
        rng : RandomNumberGenerator = default_rng,
        name=None,
        initial_value=None
    ) -> Sequence[Element]:
        mean = norm.rvs(loc=self.prior_mean, scale=self.prior_sd, size=self.size, random_state=rng.np)
        x = norm.rvs(loc=mean, scale=self.sd, random_state=rng.np)
        if self.size == 1 and isinstance(x, Sequence):
            return x[0]
        return x

    def log_probabilities(self, element : Sequence[Element]) -> float:
        marginal_sd = (self.prior_sd**2 + self.sd**2)**.5
        return norm.logpdf(element, loc=self.prior_mean, scale=marginal_sd)

    def log_probability(self, element : Sequence[Element]) -> float:
        logprobs = self.log_probabilities(element)
        if isinstance(logprobs, float):
            return logprobs
        return sum(logprobs)

    #gets posterior predictive term
    def update(self, data : Sequence[Element]) -> "NormalNormal":
        if isinstance(data, (float, int)):
            total = data
            n_datapoints = 1
        elif isinstance(data[0], (float, int)):
            total = sum(data)
            n_datapoints = len(data)
        else:
            raise ValueError(f"Invalid data shape {data}")
        new_prior_var = 1/(1/self.prior_sd**2 + n_datapoints/self.sd**2)
        new_prior_mean = (self.prior_mean/self.prior_sd + total/self.sd) * new_prior_var
        return NormalNormal(prior_mean=new_prior_mean, prior_sd=new_prior_var**.5, sd=self.sd, size=self.size)

class MultivariateNormalNormal(Multivariate):
    def __init__(self, *, prior_means : Sequence[float] = (0,),
                 prior_cov: Sequence[Sequence[float]] = ((1,),),
                 cov: Sequence[Sequence[float]] = ((1,),),
                 size=1):
        self.prior_means = np.array(prior_means)
        self.prior_cov = np.array(prior_cov)
        self.cov = np.array(cov)
        assert isinstance(size, int)
        assert len(np.shape(self.prior_cov))==2 #make sure cov is 2d
        assert len(np.shape(self.cov))==2 #make sure cov is 2d
        assert np.shape(self.prior_means)[0] == np.shape(self.prior_cov)[0] == np.shape(self.prior_cov)[1] == np.shape(self.cov)[0] == np.shape(self.cov)[1]
        self.size = size

    @property
    def element_shape(self):
        return np.shape(self.prior_means)[0]

    def __repr__(self) -> str:
        return f"MultivariateNormal(prior_means={self.prior_means}, prior_cov={self.prior_cov}, cov={self.cov}, size={self.size})"

    def sample(
        self,
        rng : RandomNumberGenerator = default_rng,
        name=None,
        initial_value=None
    ) -> Sequence[Element]:
        means = multivariate_normal.rvs(self.prior_means, self.prior_cov, size= self.size, random_state=rng.np)
        x =  np.stack([multivariate_normal.rvs(m, self.cov, random_state=rng.np) for m in means]) #stack turns from list or arrays -> matrix
        return x

    def log_probabilities(self, element : Sequence[Element]) -> float:
        # Reference: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        marginal_cov = self.prior_cov + self.cov
        return multivariate_normal.logpdf(element, mean=self.prior_means, cov=marginal_cov)

    def log_probability(self, element : Sequence[Element]) -> float:
        assert len(element) == self.element_shape or len(element[0]) == self.element_shape
        logprobs = self.log_probabilities(element)
        if isinstance(logprobs, float):
            return logprobs
        else:
            return sum(logprobs)

    def update(self, data : Sequence[Element]) -> "MultivariateNormalNormal":
        if isinstance(data[0], (float, int)):
            total = data
            n_datapoints = 1
        elif isinstance(data[0][0], (float, int)):
            total = np.sum(data,axis=0)
            n_datapoints = len(data)
        else:
            raise ValueError(f"Invalid data shape {data}")
        prior_cov_inverted = np.linalg.inv(self.prior_cov)
        cov_inverted = np.linalg.inv(self.cov)

        new_prior_cov = np.linalg.inv(prior_cov_inverted + n_datapoints * cov_inverted)
        new_prior_means = new_prior_cov @ (cov_inverted @ total + prior_cov_inverted @ self.prior_means)
        return MultivariateNormalNormal(prior_means=new_prior_means, prior_cov=new_prior_cov, cov=self.cov, size=self.size)

class MultivariateNormal(Distribution):
    def __init__(self, means=(0,), covariance=((1,),)):
        assert len(means) == len(covariance), "Means and covariance must have the same length"
        assert all(len(row) == len(means) for row in covariance), "Covariance must be a square matrix matching the length of means"
        self.means = means
        self.covariance = covariance

    dim = property(lambda self: len(self.means))
    support = cached_property(lambda self: sp.Reals**self.dim)

    def sample(self, rng=default_rng, name=None, initial_value=None) -> float:
        x = multivariate_normal.rvs(mean=self.means, cov=self.covariance, random_state=rng.np)
        return tuple(x)

    def log_probability(self, element):
        if any(isinstance(x, sp.Basic) for x in [element, self.means, self.covariance]):
            raise NotImplementedError("Symbolic computation not supported for MultivariateNormal")
        return multivariate_normal.logpdf(element, mean=self.means, cov=self.covariance)

class InverseWishart(Distribution):
    def __init__(self, df=1, scale_matrix=((1,),)):
        scale_matrix = np.array(scale_matrix)
        assert scale_matrix.ndim == 2, "Scale matrix must be a 2D array"
        assert scale_matrix.shape[0] == scale_matrix.shape[1], \
            "Scale matrix must be square"
        assert df >= scale_matrix.shape[0], \
            "Degrees of freedom must be at least the dimension of the scale matrix"
        self.df = df
        self.scale_matrix = scale_matrix

    dim = property(lambda self: len(self.scale_matrix))
    support = cached_property(lambda self: (sp.Reals**self.dim)**self.dim)

    def sample(self, rng=default_rng, name=None, initial_value=None) -> float:
        x = invwishart.rvs(df=self.df, scale=self.scale_matrix, random_state=rng.np)
        if isinstance(x, (int, float)):
            return x
        return tuple(tuple(xi) for xi in x)

    def log_probability(self, element):
        if any(isinstance(x, sp.Basic) for x in [element, self.df, self.scale_matrix]):
            raise NotImplementedError("Symbolic computation not supported for MultivariateNormal")
        return invwishart.logpdf(x, df=self.df, scale=self.scale_matrix)

    def expected_value(self, func = None):
        assert func is None, "Arbitrary expected value function not implemented for InverseWishart"
        if self.df <= self.dim + 1:
            raise ValueError("Expected value is not defined for df <= dim + 1")
        return self.scale_matrix / (self.df - self.dim - 1)
