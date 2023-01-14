import math
import random
from itertools import product
from gorgo.distributions import Categorical, Bernoulli, Multinomial, DirichletMultinomial
from gorgo.tools import isclose

def test_distribution_isclose():
    # Same distribution.
    a = Categorical(range(3))
    b = Categorical(range(3))
    assert a.isclose(b)
    assert b.isclose(a)

    # Mismatched probabilities.
    a = Categorical(range(3))
    b = Categorical(range(3), probabilities=[.1, .4, .5])
    assert not a.isclose(b)
    assert not b.isclose(a)

    # Mismatched support, but probabilities match on non-zero support.
    a = Categorical(range(2))
    b = Categorical(range(3), probabilities=[.5, .5, 0])
    assert a.isclose(b)
    assert b.isclose(a)

    # Mismatched support.
    a = Categorical(range(2))
    b = Categorical(range(3))
    assert not a.isclose(b)
    assert not b.isclose(a)

    # Different distributions, but match.
    a = Categorical([True, False])
    b = Bernoulli(0.5)
    assert a.isclose(b)
    assert b.isclose(a)

def test_Multinomial_pdf():
    balls_bins = [
        (5, 6), (1, 5), (3, 3)
    ]
    for balls, bins in balls_bins:
        support = list(product(range(balls + 1), repeat=bins))
        dist = Multinomial(categorical_support=range(bins), trials=balls)
        tot = sum([math.exp(dist.log_probability(v)) for v in support])
        assert isclose(1.0, tot)

def test_DirichletMultinomial_pdf():
    balls_bins = [
        (5, 6), (1, 5), (3, 3)
    ]
    for balls, bins in balls_bins:
        support = list(product(range(balls + 1), repeat=bins))
        dist = DirichletMultinomial(
                trials=balls,
                alphas=[random.random()*1 for _ in range(bins)]
            )
        prob = lambda v : math.exp(dist.log_probability(v))
        tot = sum([prob(v) for v in support])
        assert isclose(1.0, tot), (tot, (balls, bins))
