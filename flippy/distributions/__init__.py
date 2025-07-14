from flippy.distributions.base import Distribution, Element
from flippy.distributions.random import RandomNumberGenerator, default_rng
from flippy.distributions.builtin_dists import *

class ZeroDistributionError(Exception):
    pass
