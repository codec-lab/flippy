import math
import pytest
import numpy as np
from flippy.distributions.scipy_dists import Uniform, NormalNormal, Normal, \
    MultivariateNormalNormal, MultivariateNormal, InverseWishart
from flippy.distributions.builtin_dists import Bernoulli, Categorical
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
    hyper_mu, hyper_sigma = -1, 1
    obs = [-.75]*10
    sigma = 1
    def normal_model():
        mu = Normal(hyper_mu, hyper_sigma).sample(name='mu')
        Normal(mu, sigma).observe(obs)
        return mu

    seed = 2391299
    lw_res = LikelihoodWeighting(
        function=normal_model,
        samples=2000,
        seed=seed
    ).run()
    nn = NormalNormal(prior_mean=hyper_mu, prior_sd=hyper_sigma, sd=sigma)

    assert isclose(lw_res.expected_value(), nn.update(obs).prior_mean, atol=.01)


def test_multivariate_normal_multivariate_normal():
    mean = default_rng.random()
    priorvar = default_rng.random()
    sigma2 = default_rng.random()

    mvn = MultivariateNormalNormal(prior_means=[mean,mean],prior_cov=[[priorvar,0],[0,priorvar]],cov=[[sigma2,0],[0,sigma2]],size=3)

    samples = mvn.sample()
    uvn = NormalNormal(prior_mean=mean, prior_sd=priorvar**.5, sd=sigma2**.5,size=3)
    uvnlogprob = uvn.log_probability(samples.flatten())
    mvnlogprob = mvn.log_probability(samples)

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
