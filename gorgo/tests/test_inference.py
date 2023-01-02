import math
from gorgo import _distribution_from_inference
from gorgo.core import Bernoulli, Distribution, Categorical
from gorgo.inference import SamplePrior, Enumeration, LikelihoodWeighting, MetropolisHastings
from gorgo.inference.metropolis_hastings import Entry
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
    assert rv.keys() == set(range(1, 101))
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

def test_metropolis_hastings():
    param = 0.98
    expected = 1/param

    seed = 13842

    mh_dist = MetropolisHastings(geometric, samples=1000, burn_in=0, thinning=5, seed=seed).run(param)
    mh_exp = expectation(_distribution_from_inference(mh_dist))
    assert isclose(expected, mh_exp, atol=1e-2), 'Should be somewhat close to expected value'

def _db_from_trace(func, *, args=(), kwargs={}, trace=[]):
    ps = CPSInterpreter().initial_program_state(func)
    ps = ps.step(*args, **kwargs)
    db = {}
    lp = 0
    for dist, value in trace:
        assert ps.distribution.isclose(dist)
        if isinstance(ps, SampleState):
            log_prob = ps.distribution.log_probability(value)
            lp += log_prob
            db[ps.name] = Entry(
                ps.name, ps.distribution, value, log_prob, True
            )
            ps = ps.step(value)
        else:
            assert False, f'Unexpected state {ps}'
    assert isinstance(ps, ReturnState)
    return db, lp

def test_mh_acceptance_ratio():
    def fn():
        if Bernoulli(0.5).sample(name='choice'):
            return Categorical(range(2)).sample(name='rv')
        else:
            return Categorical(range(3)).sample(name='rv')

    db, lp = _db_from_trace(fn, trace=[
        (Bernoulli(0.5), 0),
        (Categorical(range(3)), 1),
    ])

    new_db, new_lp = _db_from_trace(fn, trace=[
        (Bernoulli(0.5), 1),
        (Categorical(range(2)), 1),
    ])

    # Only need score difference since proposal probabilities are the same.
    # They're the same b/c we have the same # of variables and the resampled variable
    # is uniform.
    acceptance_ratio = new_lp - lp

    mh = MetropolisHastings(None, None)
    assert isclose(mh.calc_log_acceptance_ratio("choice", new_db, db), acceptance_ratio)
    assert isclose(mh.calc_log_acceptance_ratio("choice", db, new_db), -acceptance_ratio)

def test_single_site_mh():
    def fn():
        if Bernoulli(.5).sample(name="choice"):
            x = Categorical(['a', 'b'], probabilities=[.5, .5]).sample(name='x')
        else:
            x = Categorical(['c', 'b'], probabilities=[.8, .2]).sample(name='x')
        return x
    enum_dist = Enumeration(fn).run()
    mh_dist = MetropolisHastings(fn, samples=10000, seed=124).run()
    for e in enum_dist:
        assert isclose(enum_dist[e], mh_dist[e], atol=1e-2)