import math
from gorgo.distributions.scipy_dists import Uniform, NormalNormal, Normal
from gorgo.inference.likelihood_weighting import LikelihoodWeighting
from gorgo.tools import isclose

def test_scipy_uniform():
    dist = Uniform(-1, -.5)
    for i in range(100):
        u = dist.sample()
        assert -1 <= u <= -.5
        assert isclose(dist.log_probability(u), math.log(1/.5))

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
