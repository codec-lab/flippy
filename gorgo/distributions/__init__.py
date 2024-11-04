from gorgo.distributions.base import Distribution, Element
from gorgo.distributions.random import RandomNumberGenerator, default_rng
from gorgo.distributions.builtin_dists import *

class ZeroDistributionError(Exception):
    pass
