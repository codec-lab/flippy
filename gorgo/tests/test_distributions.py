import math
import pytest
from gorgo.distributions.scipy_dists import Uniform, NormalNormal, Normal, MultivariateNormal
from gorgo.inference.likelihood_weighting import LikelihoodWeighting
from gorgo.tools import isclose
from gorgo.distributions.random import default_rng

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

    mvn = MultivariateNormal(prior_means=[mean,mean],prior_cov=[[priorvar,0],[0,priorvar]],cov=[[sigma2,0],[0,sigma2]],size=3)

    samples = mvn.sample()
    uvn = NormalNormal(prior_mean=mean, prior_sd=priorvar**.5, sd=sigma2**.5,size=3)
    uvnlogprob = uvn.log_probability(samples.flatten())
    mvnlogprob = mvn.log_probability(samples)

    assert isclose(uvnlogprob, mvnlogprob)
