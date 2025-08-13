"""
Distributions are used to represent random variables in probabilistic models.
FlipPy provides several built-in distributions, and new distributions can be
defined by subclassing the `Distribution` base class.
"""

from flippy.distributions.base import Distribution, Element
from flippy.distributions.random import RandomNumberGenerator, default_rng
from flippy.distributions.builtin_dists import Bernoulli, Categorical, Multinomial, \
    Gaussian, Normal, Gamma, Uniform, Beta, Binomial, Geometric, Poisson, \
    BetaBinomial, Dirichlet, DirichletMultinomial, Mixture
from flippy.distributions.scipy_dists import MultivariateNormal, InverseWishart, \
    NormalNormal, MultivariateNormalNormal

__all__ = [
    "RandomNumberGenerator",
    "Distribution",

    "Bernoulli",
    "Categorical",
    "Multinomial",
    "Gaussian",
    "Normal",
    "Gamma",
    "Uniform",
    "Beta",
    "Binomial",
    "Geometric",
    "Poisson",
    "BetaBinomial",
    "Dirichlet",
    "DirichletMultinomial",
    "NormalNormal",
    "MultivariateNormal",
    "InverseWishart",
    "MultivariateNormalNormal",

    "Mixture"
]

class ZeroDistributionError(Exception):
    pass
