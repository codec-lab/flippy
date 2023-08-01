import math
from gorgo import _distribution_from_inference
from gorgo.distributions import Bernoulli, Distribution, Categorical, Dirichlet, Normal, Gamma, Uniform
from gorgo.inference import SamplePrior, Enumeration, LikelihoodWeighting
from gorgo.tools import isclose
from gorgo.interpreter import CPSInterpreter, ReturnState, SampleState, ObserveState

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

import numpy as np
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return (a@b)/((a**2).sum() * (b**2).sum())**.5

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
