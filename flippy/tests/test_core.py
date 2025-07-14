import math
import random
from itertools import product
from flippy.distributions.builtin_dists import Categorical, Bernoulli, \
    Multinomial, DirichletMultinomial, Dirichlet, Multinomial, Beta, \
    Binomial, BetaBinomial
from flippy.inference import LikelihoodWeighting
from flippy.distributions.random import RandomNumberGenerator
from flippy.tools import isclose

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

    # Matched with an absolute tolerance
    a = Bernoulli(0.5 + 1e-10)
    b = Bernoulli(0.5)
    assert a.isclose(b)

    # Tolerance can be changed
    a = Bernoulli(0.50001)
    b = Bernoulli(0.5)
    assert not a.isclose(b)
    assert not a.isclose(b, atol=1e-5)
    assert a.isclose(b, atol=1e-4)
    assert a.isclose(b, atol=1e-3)

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

def test_dirichletmultinomial_update():
    data = [(0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1)]

    def model():
        p = Dirichlet([1, 1, 1]).sample()
        dist = Multinomial(categorical_support=range(len(p)), trials=1, probabilities=p)
        [dist.observe(d) for d in data]
        return dist.sample()

    seed = 139491
    res = LikelihoodWeighting(model, samples=2000, seed=seed).run()

    dist = DirichletMultinomial(trials=1, alphas=[1, 1, 1])
    dist2 = dist.update(data)

    for e in dist2.support:
        assert isclose(dist2.prob(e), res.prob(e), atol=.01)

def test_betabinomial_update():
    data = [0]*3 + [1]*2 + [2]*1

    def model():
        p = Beta(1, 2).sample()
        dist = Binomial(trials=3, p=p)
        [dist.observe(d) for d in data]
        return dist.sample()

    seed = 139421
    res = LikelihoodWeighting(model, samples=10000, seed=seed).run()

    dist = BetaBinomial(trials=3, alpha=1, beta=1)
    dist2 = dist.update(data)
    for e in dist2.support:
        assert isclose(dist2.prob(e), res.prob(e), atol=.05)


def test_random_number_generation():
    rng1a = RandomNumberGenerator(20)
    rng1b = RandomNumberGenerator(20)
    rng1c = RandomNumberGenerator(20)
    assert rng1a.randint(0, 100) == rng1b.randint(0, 100)
    assert rng1a != rng1b != rng1c
    assert rng1a.randint(0, 100) != rng1c.randint(0, 100)
    rng2 = RandomNumberGenerator(30)
    rng3 = RandomNumberGenerator(40)
    assert rng2.randint(0, 100) != rng3.randint(0, 100)

def test_random_number_numpy():
    rng1 = RandomNumberGenerator(30)
    rng2 = RandomNumberGenerator(30)
    rng3 = RandomNumberGenerator(40)
    assert rng1.np.random() == rng2.np.random()
    assert rng1.np.random() != rng3.np.random()
