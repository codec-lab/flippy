import math
from gorgo import _distribution_from_inference
from gorgo.distributions import Bernoulli, Distribution, Categorical, Dirichlet, Normal, Gamma, Uniform
from gorgo.inference import SamplePrior, Enumeration, LikelihoodWeighting
from gorgo.inference.metropolis_hastings import Entry
from gorgo.tools import isclose
from gorgo.interpreter import CPSInterpreter, ReturnState, SampleState, ObserveState
from gorgo.inference.metropolis_hastings import Mapping, Hashable
import dataclasses

def geometric(p):
    '''
    The probability distribution of the number X of Bernoulli trials needed to get one success.
    https://en.wikipedia.org/wiki/Geometric_distribution
    '''
    x = Bernoulli(p).sample()
    if x == 1:
        return 1
    return 1 + geometric(p)

def expectation(d: Distribution, projection=lambda s: s):
    total = 0
    partition = 0
    for s in d.support:
        p = math.exp(d.log_probability(s))
        total += p * projection(s)
        partition += p
    assert isclose(partition, 1)
    return total

def test_enumeration_geometric():
    param = 0.25
    expected = 1/param
    rv = Enumeration(geometric, max_executions=100).run(param)
    d = _distribution_from_inference(rv)
    assert isclose(expectation(d), expected)

    assert len(rv) == 100
    assert set(rv.keys()) == set(range(1, 101)), set(rv.keys()) - set(range(1, 101))
    for k, sampled_prob in rv.items():
        pmf = (1-param) ** (k - 1) * param
        # This will only be true when executions is high enough, since
        # sampled_prob is normalized.
        assert isclose(sampled_prob, pmf), (k, sampled_prob, pmf)

def test_likelihood_weighting_and_sample_prior():
    param = 0.98
    expected = 1/param

    seed = 13842

    lw_dist = LikelihoodWeighting(geometric, samples=1000, seed=seed).run(param)
    lw_exp = expectation(_distribution_from_inference(lw_dist))
    prior_dist = SamplePrior(geometric, samples=1000, seed=seed).run(param)
    prior_exp = expectation(_distribution_from_inference(prior_dist))

    assert lw_exp == prior_exp, 'Should be identical when there are no observe statements'

    assert isclose(expected, prior_exp, atol=1e-2), 'Should be somewhat close to expected value'

# def test_metropolis_hastings():
#     param = 0.98
#     expected = 1/param

#     seed = 13842

#     mh_dist = MetropolisHastings(geometric, samples=1000, burn_in=0, thinning=5, seed=seed).run(param)
#     mh_exp = expectation(_distribution_from_inference(mh_dist))
#     assert isclose(expected, mh_exp, atol=1e-2), 'Should be somewhat close to expected value'

import numpy as np
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return (a@b)/((a**2).sum() * (b**2).sum())**.5

# def test_metropolis_hastings_dirichlet_categorical():
#     c1_params = [1, 1, 1]
#     c1_data = list('ababababacc')*2
#     def model():
#         c1 = Dirichlet(c1_params).sample(name='c1')
#         dist1 = Categorical(support=list('abc'), probabilities=c1)
#         [dist1.observe(d) for d in c1_data]
#         return c1
#     seed = 13842
#     exp_c1 = [n + sum([d == c for d in c1_data]) for c, n in zip('abc', c1_params)]
#     exp_c1 = [n / sum(exp_c1) for n in exp_c1]

#     mh_params = dict(
#         function=model,
#         samples=1000,
#         burn_in=500,
#         thinning=2,
#         seed=seed
#     )

#     # test without/with drift kernel
#     mh_dist = MetropolisHastings(
#         **mh_params,
#         uniform_drift_kernel_width=None,
#     ).run()
#     est_c1 = mh_dist.expected_value(lambda c1: np.array(c1))
#     assert cosine_similarity(exp_c1, est_c1) > .99

#     mh_dist = MetropolisHastings(
#         **mh_params,
#         uniform_drift_kernel_width=.15,
#     ).run()
#     est_c1 = mh_dist.expected_value(lambda c1: np.array(c1))
#     assert cosine_similarity(exp_c1, est_c1) > .99

# def test_metropolis_hastings_normal_normal():
#     hyper_mu, hyper_sigma = 1.4, 2
#     obs = [-.75]
#     sigma = 1
#     def normal_model():
#         mu = Normal(hyper_mu, hyper_sigma).sample(name='mu')
#         Normal(mu, sigma).observe(obs)
#         return mu

#     seed = 2191299
#     new_sigma = 1/(1/(hyper_sigma**2) + len(obs)/(sigma**2))
#     new_mu = (hyper_mu/(hyper_sigma**2) + sum(obs)/(sigma**2))*new_sigma

#     mh_dist = MetropolisHastings(normal_model, samples=20000, burn_in=0, thinning=1, seed=seed).run()
#     mh_dist = _distribution_from_inference(mh_dist)
#     mh_exp = expectation(mh_dist)
#     assert isclose(new_mu, mh_exp, atol=1e-2), (new_mu, mh_exp)

# def test_metropolis_hastings_gamma():
#     def gamma_model():
#         g = Gamma(3, 2).sample()
#         Uniform(0, g).observe(0)
#         return g

#     mh_dist = MetropolisHastings(gamma_model, samples=10000, burn_in=0, thinning=1, seed=38837).run()
#     lw_dist = LikelihoodWeighting(gamma_model, samples=10000, seed=18837).run()
#     assert isclose(expectation(mh_dist), expectation(lw_dist), rtol=.05)

def test_observations():
    def model_simple():
        rv = Categorical(range(3)).sample()
        Bernoulli(2**(-rv)).observe(True)
        return rv

    def model_branching():
        if Bernoulli(0.5).sample(name='choice'):
            Bernoulli(.2).observe(True, name='obs')
            return Categorical(range(2)).sample(name='rv')
        else:
            return Categorical(range(3)).sample(name='rv')

    seed = 13842
    samples = 5000

    for model, expected_dist in [
        (
            model_simple,
            Categorical(range(3), probabilities=[4/7, 2/7, 1/7]),
        ),
        (
            model_branching,
            Categorical(range(3), probabilities=[
                1/6 * 1/2 + 5/6 * 1/3,
                1/6 * 1/2 + 5/6 * 1/3,
                5/6 * 1/3,
            ]),
        ),
    ]:
        print('model', model)

        dist = _distribution_from_inference(Enumeration(model).run())
        print('Enumeration', dist)
        assert dist.isclose(expected_dist)

        dist = _distribution_from_inference(LikelihoodWeighting(model, samples=samples, seed=seed).run())
        print('LikelihoodWeighting', dist)
        assert dist.isclose(expected_dist, atol=1e-1)
