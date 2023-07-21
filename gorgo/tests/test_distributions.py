import math
from gorgo.distributions.scipy_dists import Uniform
from gorgo.tools import isclose

def test_scipy_uniform():
    dist = Uniform(-1, -.5)
    for i in range(100):
        u = dist.sample()
        assert -1 <= u <= -.5
        assert isclose(dist.log_probability(u), math.log(1/.5))
