from typing import Sequence
from functools import cached_property

from flippy.distributions.base import Distribution, Element
from flippy.distributions.random import RandomNumberGenerator, default_rng
from flippy.tools import isclose, PackagePlaceholder

try:
    import scipy.stats as scipy_stats
except ImportError:
    scipy_stats = PackagePlaceholder("scipy.stats")
try:
    import numpy as np
except ImportError:
    np = PackagePlaceholder("numpy")
try:
    import sympy as sp
except ImportError:
    sp = PackagePlaceholder("sympy")


class MultivariateNormal(Distribution):
    """Multivariate normal distribution with given means and covariance matrix."""
    def __init__(self, means=(0,), covariance=((1,),)):
        assert len(means) == len(covariance), "Means and covariance must have the same length"
        assert all(len(row) == len(means) for row in covariance), "Covariance must be a square matrix matching the length of means"
        self.means = means
        self.covariance = covariance

    dim = property(lambda self: len(self.means))
    support = cached_property(lambda self: sp.Reals**self.dim)

    def sample(self, rng=default_rng, name=None, initial_value=None) -> float:
        x = scipy_stats.multivariate_normal.rvs(mean=self.means, cov=self.covariance, random_state=rng.np)
        if isinstance(x, (int, float)):
            return (x,)
        return tuple(x)

    def log_probability(self, element):
        if any(isinstance(x, sp.Basic) for x in [element, self.means, self.covariance]):
            raise NotImplementedError("Symbolic computation not supported for MultivariateNormal")
        return scipy_stats.multivariate_normal.logpdf(element, mean=self.means, cov=self.covariance)

class InverseWishart(Distribution):
    """Inverse Wishart distribution with given degrees of freedom and scale matrix."""
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
        x = scipy_stats.invwishart.rvs(df=self.df, scale=self.scale_matrix, random_state=rng.np)
        if isinstance(x, (int, float)):
            return x
        return tuple(tuple(xi) for xi in x)

    def log_probability(self, element):
        if any(isinstance(x, sp.Basic) for x in [element, self.df, self.scale_matrix]):
            raise NotImplementedError("Symbolic computation not supported for MultivariateNormal")
        return scipy_stats.invwishart.logpdf(element, df=self.df, scale=self.scale_matrix)

    def expected_value(self, func = None):
        assert func is None, "Arbitrary expected value function not implemented for InverseWishart"
        if self.df <= self.dim + 1:
            raise ValueError("Expected value is not defined for df <= dim + 1")
        return self.scale_matrix / (self.df - self.dim - 1)


class NormalNormal(Distribution):
    """
    Compound distribution for a normal distribution
    with unknown mean and known variance. The prior on the mean is
    also a normal distribution, which is its own conjugate distribution.
    """
    def __init__(self, *, prior_mean=0, prior_sd=1, sd=1):
        self.prior_mean = np.array(prior_mean)
        self.prior_sd = np.array(prior_sd)
        self.sd = np.array(sd)
    marginal_sd = property(lambda self: (self.prior_sd**2 + self.sd**2)**.5)

    def sample(
        self,
        rng : RandomNumberGenerator = default_rng,
        name=None,
        initial_value=None
    ) -> Sequence[Element]:
        #first sample mu and then sample x
        #see scipy norm documentation for information on loc and scale
        mean = scipy_stats.norm.rvs(loc=self.prior_mean, scale=self.prior_sd, random_state=rng.np)
        x = scipy_stats.norm.rvs(loc=mean, scale=self.sd, random_state=rng.np)
        return x

    def log_probability(self, element : Sequence[Element]) -> float:
        return scipy_stats.norm.logpdf(element, loc=self.prior_mean, scale=self.marginal_sd)

    def update(self, data : Element | Sequence[Element]) -> "NormalNormal":
        #gets posterior predictive term
        if isinstance(data, (float, int)):
            data = [data]  # Ensure data is a sequence
        total = sum(data)
        n_datapoints = len(data)
        new_prior_var = 1/(1/self.prior_sd**2 + n_datapoints/self.sd**2)
        new_prior_sd = new_prior_var**.5
        new_prior_mean = (self.prior_mean/self.prior_sd**2 + total/self.sd**2) * new_prior_var
        return NormalNormal(prior_mean=new_prior_mean, prior_sd=new_prior_sd, sd=self.sd)

class MultivariateNormalNormal(Distribution):
    """
    Compound distribution for a multivariate normal distribution
    with unknown mean and known covariance. The prior on the mean is
    also a multivariate normal distribution, which is its own conjugate distribution.
    """
    def __init__(
        self,
        *,
        prior_means : Sequence[float] = (0,),
        prior_cov: Sequence[Sequence[float]] = ((1,),),
        cov: Sequence[Sequence[float]] = ((1,),),
    ):
        self.prior_means = np.array(prior_means)
        self.prior_cov = np.array(prior_cov)
        self.cov = np.array(cov)
        assert len(np.shape(self.prior_cov)) == 2 #make sure cov is 2d
        assert len(np.shape(self.cov)) == 2 #make sure cov is 2d
        assert np.shape(self.prior_means)[0] == np.shape(self.prior_cov)[0] == \
            np.shape(self.prior_cov)[1] == np.shape(self.cov)[0] == \
            np.shape(self.cov)[1], "Means and covariances must match in dimensions"

    def sample(
        self,
        rng : RandomNumberGenerator = default_rng,
        name=None,
        initial_value=None
    ) -> Sequence[Element]:
        mean = scipy_stats.multivariate_normal.rvs(self.prior_means, self.prior_cov, random_state=rng.np)
        x = scipy_stats.multivariate_normal.rvs(mean, self.cov, random_state=rng.np)
        return x

    def log_probabilities(self, element : Sequence[Element]) -> float:
        # Reference: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        element = np.array(element)
        assert np.ndim(element) in (1, 2), "Element must be a 1D or 2D array"
        if np.ndim(element) == 1:
            element = element.reshape(1, -1)
        assert np.shape(element)[1] == len(self.prior_means), \
            f"Each element must have shape {len(self.prior_means)}, got {np.shape(element)}"
        marginal_cov = self.prior_cov + self.cov
        return scipy_stats.multivariate_normal.logpdf(element, mean=self.prior_means, cov=marginal_cov)

    def log_probability(self, element : Sequence[Element]) -> float:
        return self.log_probabilities(element)

    def update(self, data : Sequence[Element]) -> "MultivariateNormalNormal":
        raise NotImplementedError
        # if isinstance(data[0], (float, int)):
        #     total = data
        #     n_datapoints = 1
        # elif isinstance(data[0][0], (float, int)):
        #     total = np.sum(data,axis=0)
        #     n_datapoints = len(data)
        # else:
        #     raise ValueError(f"Invalid data shape {data}")
        # prior_cov_inverted = np.linalg.inv(self.prior_cov)
        # cov_inverted = np.linalg.inv(self.cov)

        # new_prior_cov = np.linalg.inv(prior_cov_inverted + n_datapoints * cov_inverted)
        # new_prior_means = new_prior_cov @ (cov_inverted @ total + prior_cov_inverted @ self.prior_means)
        # return MultivariateNormalNormal(prior_means=new_prior_means, prior_cov=new_prior_cov, cov=self.cov, size=self.size)
