import math

import pytest
import numpy as np

from flippy.distributions import Bernoulli, Categorical, Dirichlet, Uniform, \
    Normal, Gamma, Beta, MultivariateNormal, InverseWishart, NormalNormal, \
    MultivariateNormalNormal
from flippy.distributions.support import Interval, Range, ProductSet, \
    UnionSet, Simplex, OrderedIntegerPartitions
from flippy.inference.likelihood_weighting import LikelihoodWeighting
from flippy.inference.enumeration import Enumeration
from flippy.interpreter import CPSInterpreter
from flippy.core import ReturnState
from flippy.tools import isclose
from flippy.distributions.random import default_rng, RandomNumberGenerator

def test_scipy_uniform():
    dist = Uniform(-1, -.5)
    for i in range(100):
        u = dist.sample()
        assert -1 <= u <= -.5
        assert isclose(dist.log_probability(u), math.log(1/.5))

def test_distribution_bool():
    dist = Uniform(-1, -.5)
    with pytest.raises(ValueError):
        bool(dist)

def test_normal_normal():
    hyper_mu, hyper_sigma = -1, 4
    obs = [-.75]*10
    sigma = 1
    def normal_model():
        mu = Normal(hyper_mu, hyper_sigma).sample(name='mu')
        dist = Normal(mu, sigma)
        dist.observe_all(obs)
        return mu

    seed = 2391299
    lw_res = LikelihoodWeighting(
        function=normal_model,
        samples=4000,
        seed=seed
    ).run()
    nn = NormalNormal(prior_mean=hyper_mu, prior_sd=hyper_sigma, sd=sigma)

    assert isclose(lw_res.expected_value(), nn.update(obs).prior_mean, atol=.01)


def test_multivariate_normal_multivariate_normal():
    mean = default_rng.random()
    priorvar = default_rng.random()
    sigma2 = default_rng.random()

    mvn = MultivariateNormalNormal(
        prior_means=[mean,mean],
        prior_cov=[[priorvar,0],[0,priorvar]],
        cov=[[sigma2,0],[0,sigma2]]
    )

    sample = mvn.sample()
    uvn = NormalNormal(prior_mean=mean, prior_sd=priorvar**.5, sd=sigma2**.5)
    uvnlogprob = uvn.total_log_probability(sample.flatten())
    mvnlogprob = mvn.log_probability(sample)

    assert isclose(uvnlogprob, mvnlogprob)


def test_multivariate_normal():
    rng = RandomNumberGenerator(12345)
    mvn = MultivariateNormal(means=(0, 1), covariance=((1, 0.5), (0.5, 1)))
    samples = np.array([mvn.sample(rng=rng) for _ in range(5000)])
    sample_means = np.mean(samples, axis=0)
    sample_cov = np.prod(samples - sample_means, axis=1).mean()
    assert (np.abs(sample_means - (0, 1)) < .02).all()
    assert abs(sample_cov - 0.5) < 1e-2


def test_InverseWishart():
    invwish = InverseWishart(df=4.2, scale_matrix=((1, 0.5), (0.5, 1)))
    rng = RandomNumberGenerator(32345)
    samples = [invwish.sample(rng=rng) for _ in range(5000)]
    assert np.isclose(
        np.array(samples).mean(axis=0),
        invwish.expected_value(),
        atol=0.1
    ).all()

def test_categorical_dist_equality():
    def f():
        if Bernoulli().sample():
            return Categorical(['A', 'B'])
        return Categorical(['A', 'B'])
    dist = Enumeration(f).run()
    assert dist.isclose(Categorical([
        Categorical(['A', 'B'], probabilities=[.5, .5])
    ]))

def test_categorical_condition():
    d = Categorical(range(10))
    d = d.condition(lambda x : x % 2 == 0)
    d = d.condition(lambda x : x % 3 == 0)
    assert d.isclose(Categorical([0, 6], probabilities=[.5, .5]))

def test_observe_all():
    def model_observe_all(p, data):
        Bernoulli(p).observe_all(data)
        return p

    def model_observe(p, data):
        [Bernoulli(p).observe(d) for d in data]
        return p

    data = (1, 1, 1, 1, 1, 0, 0)
    p = .9

    ps = CPSInterpreter().initial_program_state(model_observe_all)
    ps = ps.step(p, data)
    logprob1 = ps.distribution.log_probability(ps.value)

    ps = CPSInterpreter().initial_program_state(model_observe)
    ps = ps.step(p, data)
    logprob2 = 0
    while not isinstance(ps, ReturnState):
        logprob2 += ps.distribution.log_probability(ps.value)
        ps = ps.step()
    assert isclose(logprob1, logprob2)

def test_Dirichlet_numerical_stability():
    rng = RandomNumberGenerator(12345)
    for _ in range(100):
        dist = Dirichlet([.1, .0001, 4.0])
        p = dist.sample(rng=rng)
        assert 0 not in p, "We should not be able to sample a vector with a 0"
        assert 1 not in p, "We should not be able to sample a vector with a 1"
        assert dist.log_probability(p) > float('-inf')

def test_support():
    closed_interval = Interval(0, 1, left_open=False, right_open=False)
    assert 0.0 in closed_interval
    assert 1.0 in closed_interval

    open_interval = Interval(0, 1, left_open=True, right_open=True)
    assert .00001 in open_interval
    assert .99999 in open_interval
    assert 0.0 not in open_interval
    assert 1.0 not in open_interval

    half_open_interval = Interval(0, 1, left_open=True, right_open=False)
    assert .00001 in half_open_interval
    assert .99999 in half_open_interval
    assert 0.0 not in half_open_interval
    assert 1.0 in half_open_interval

    range1 = Range(10)
    assert 0 in range1
    assert 9 in range1
    assert 10 not in range1

    range2 = Range(5, 10)
    assert 1 not in range2
    assert 5 in range2
    assert 9 in range2
    assert 10 not in range2

    range3 = Range(5, 10, 2)
    assert 5 in range3
    assert 7 in range3
    assert 9 in range3
    assert 6 not in range3
    assert 10 not in range3
    assert 11 not in range3

    product_set_1 = ProductSet(Range(0, 2), Range(5, 7))
    assert (0, 5) in product_set_1
    assert (1, 6) in product_set_1
    assert (2, 5) not in product_set_1
    assert (0, 7) not in product_set_1
    assert (1, 5) in product_set_1
    assert 1 not in product_set_1

    product_set_2 = ProductSet(Range(10), ("A", "B", "C"))
    assert (0, "A") in product_set_2
    assert (9, "C") in product_set_2
    assert (10, "A") not in product_set_2

    product_set_3 = ProductSet(Range(0, 2), Interval(0, 2))
    assert (0, 0.5) in product_set_3
    assert (1, 1.5) in product_set_3
    assert (2, 0.5) not in product_set_3

    union_set_1 = UnionSet(Range(0, 2), Range(5, 7))
    assert 0 in union_set_1
    assert 1 in union_set_1
    assert 5 in union_set_1
    assert 6 in union_set_1
    assert 2 not in union_set_1
    assert 4 not in union_set_1

    simplex_3 = Simplex(3)
    assert (0.5, 0.5, 0.0) in simplex_3
    assert (0.3, 0.4, 0.3) in simplex_3
    assert (0.1, 0.2, 0.6) not in simplex_3
    assert (.5, 0.5, 0.1) not in simplex_3

    ordered_partitions = OrderedIntegerPartitions(5, 3)
    assert (5, 0, 0) in ordered_partitions
    assert (4, 1, 0) in ordered_partitions
    assert (3, 2, 0) in ordered_partitions
    assert (2, 1, 1) not in ordered_partitions
