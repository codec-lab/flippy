import math
from gorgo import _distribution_from_inference
from gorgo.distributions import Bernoulli, Distribution, Categorical, Dirichlet, Normal, Gamma, Uniform
from gorgo.inference import SamplePrior, Enumeration, LikelihoodWeighting, MetropolisHastings
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

def test_metropolis_hastings():
    param = 0.98
    expected = 1/param

    seed = 13842

    mh_dist = MetropolisHastings(geometric, samples=1000, burn_in=0, thinning=5, seed=seed).run(param)
    mh_exp = expectation(_distribution_from_inference(mh_dist))
    assert isclose(expected, mh_exp, atol=1e-2), 'Should be somewhat close to expected value'

def test_metropolis_hastings_dirichlet_categorical():
    def model():
        data = list('ababac'+'ccccba')
        c1 = Dirichlet([.1, .1, .1]).sample()
        c2 = Dirichlet([1, 1, 1]).sample()
        dist1 = Categorical(support=list('abc'), probabilities=c1)
        dist2 = Categorical(support=list('abc'), probabilities=c2)
        cat = tuple([0]*6 + [1]*6)
        [{0: dist1, 1: dist2}[c].observe(d) for d, c in list(zip(data, cat))]
        return tuple(c1 + c2)

    seed = 13842
    expected = [3.1/6.3, 2.1/6.3, 1.1/6.3, 2/9, 2/9, 5/9]

    mh_dist = MetropolisHastings(model, samples=5000, burn_in=0, thinning=2, seed=seed).run()
    mh_dist = _distribution_from_inference(mh_dist)
    mh_exp = [expectation(mh_dist, projection=lambda s: s[i]) for i in range(6)]
    for exp, mh_expectation in zip(expected, mh_exp):
        assert isclose(exp, mh_expectation, atol=1e-2), (exp, mh_expectation)

def test_metropolis_hastings_normal_normal():
    hyper_mu, hyper_sigma = 1.4, 2
    obs = [-.75]
    sigma = 1
    def normal_model():
        mu = Normal(hyper_mu, hyper_sigma).sample(name='mu')
        Normal(mu, sigma).observe(obs)
        return mu

    seed = 2191299
    new_sigma = 1/(1/(hyper_sigma**2) + len(obs)/(sigma**2))
    new_mu = (hyper_mu/(hyper_sigma**2) + sum(obs)/(sigma**2))*new_sigma

    mh_dist = MetropolisHastings(normal_model, samples=20000, burn_in=0, thinning=1, seed=seed).run()
    mh_dist = _distribution_from_inference(mh_dist)
    mh_exp = expectation(mh_dist)
    assert isclose(new_mu, mh_exp, atol=1e-2), (new_mu, mh_exp)

def test_metropolis_hastings_gamma():
    def gamma_model():
        g = Gamma(3, 2).sample()
        Uniform(0, g).observe(0)
        return g

    mh_dist = MetropolisHastings(gamma_model, samples=10000, burn_in=0, thinning=1, seed=38837).run()
    lw_dist = LikelihoodWeighting(gamma_model, samples=10000, seed=18837).run()
    assert isclose(expectation(mh_dist), expectation(lw_dist), rtol=.05)

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

        dist = _distribution_from_inference(MetropolisHastings(model, samples=samples, seed=seed).run())
        print('MetropolisHastings', dist)
        assert dist.isclose(expected_dist, atol=1e-1)

@dataclasses.dataclass
class DBResult:
    db: Mapping[Hashable, Entry]
    @property
    def sample_count(self):
        return sum(1 for entry in self.db.values() if entry.is_sample)
    @property
    def log_prior(self):
        return sum(entry.log_prob for entry in self.db.values() if entry.is_sample)
    @property
    def log_likelihood(self):
        return sum(entry.log_prob for entry in self.db.values() if not entry.is_sample)
    def log_proposal(self, new: 'DBResult', sample_name: str):
        '''
        Log probability of proposing `new`, when starting from `self`.
        '''
        log_prior = 0
        for new_name in (new.db.keys() - self.db.keys()) | {sample_name}:
            new_entry = new.db[new_name]
            if new_entry.is_sample:
                log_prior += new_entry.log_prob
        return math.log(1/self.sample_count) + log_prior
    def acceptance_ratio(self, new_db: 'DBResult', proposal_name: str):
        '''
        This is a verbosely-computed acceptance ratio, assuming `new_db` is proposed
        from the `self` DB.
        '''
        log_proposal_to_new = self.log_proposal(new_db, proposal_name)
        log_proposal_to_old = new_db.log_proposal(self, proposal_name)
        # These are unnormalized, but that is fine because we return a ratio
        log_probability_new = new_db.log_prior + new_db.log_likelihood
        log_probability_old = self.log_prior + self.log_likelihood
        return (
            log_probability_new + log_proposal_to_old
            - (log_probability_old + log_proposal_to_new)
        )

def _db_from_trace(func, *, args=(), kwargs={}, trace=[]):
    ps = CPSInterpreter().initial_program_state(func).step(*args, **kwargs)
    db = {}
    for dist, value in trace:
        assert ps.distribution.isclose(dist)
        if isinstance(ps, (SampleState, ObserveState)):
            is_sample = isinstance(ps, SampleState)
            if not is_sample:
                assert isclose(ps.value, value)
            log_prob = ps.distribution.log_probability(value)
            db[ps.name] = Entry(ps.name, ps.distribution, value, log_prob, is_sample)
            ps = ps.step(value) if is_sample else ps.step()
        else:
            assert False, f'Unexpected state {ps}'
    assert isinstance(ps, ReturnState), f'Did not terminate in return state, instead: {ps}'
    return DBResult(db)

def test_db_result():
    mh = MetropolisHastings(None, None)

    def fn():
        if Bernoulli(0.5).sample(name='choice'):
            return Categorical(range(2)).sample(name='rv')
        else:
            Bernoulli(.8).observe(True)
            return Categorical(range(3)).sample(name='rv')

    def _test_acceptance(new_db, db, sample_name, expected_ratio):
        assert isclose(db.acceptance_ratio(new_db, sample_name), expected_ratio)
        assert isclose(new_db.acceptance_ratio(db, sample_name), -expected_ratio)
        assert isclose(mh.calc_log_acceptance_ratio(sample_name, new_db.db, db.db), expected_ratio)
        assert isclose(mh.calc_log_acceptance_ratio(sample_name, db.db, new_db.db), -expected_ratio)

    db_result = _db_from_trace(fn, trace=[
        (Bernoulli(0.5), 0),
        (Bernoulli(.8), True),
        (Categorical(range(3)), 1),
    ])

    assert db_result.sample_count == 2
    assert db_result.log_prior == math.log(1/2 * 1/3)
    assert db_result.log_likelihood == math.log(.8)
    # Probabilities are: 1) choice among variables, then 2) choice among options for variable.
    assert db_result.log_proposal(db_result, 'choice') == math.log(1/2) + math.log(1/2)
    assert db_result.log_proposal(db_result, 'rv') == math.log(1/2) + math.log(1/3)
    _test_acceptance(db_result, db_result, 'rv', 0)

    # Trying a resampling of `rv`
    new_db_result = _db_from_trace(fn, trace=[
        (Bernoulli(0.5), 0),
        (Bernoulli(.8), True),
        (Categorical(range(3)), 2),
    ])
    # This probability is similar to above.
    assert db_result.log_proposal(new_db_result, 'rv') == math.log(1/2) + math.log(1/3)
    _test_acceptance(new_db_result, db_result, 'rv', 0)

    # Trying a resampling of `choice`
    new_db_result = _db_from_trace(fn, trace=[
        (Bernoulli(0.5), 1),
        (Categorical(range(2)), 1),
    ])
    # This probability is similar to above.
    assert db_result.log_proposal(new_db_result, 'choice') == math.log(1/2) + math.log(1/2)
    # Likelihood ratio. New trace at left, old trace at right.
    ratio = math.log(1/2) - (math.log(1/3) + math.log(.8))
    _test_acceptance(new_db_result, db_result, 'choice', ratio)

def test_mh_acceptance_ratio():
    def fn():
        if Bernoulli(0.5).sample(name='choice'):
            return Categorical(range(2)).sample(name='rv')
        else:
            return Categorical(range(3)).sample(name='rv')

    db_result = _db_from_trace(fn, trace=[
        (Bernoulli(0.5), 0),
        (Categorical(range(3)), 1),
    ])

    new_db_result = _db_from_trace(fn, trace=[
        (Bernoulli(0.5), 1),
        (Categorical(range(2)), 1),
    ])

    # Only need score difference since proposal probabilities are the same.
    # They're the same b/c we have the same # of variables and the resampled variable
    # is uniform.
    acceptance_ratio = new_db_result.log_prior - db_result.log_prior
    assert isclose(acceptance_ratio, math.log((1/2) / (1/3)))
    assert isclose(acceptance_ratio, db_result.acceptance_ratio(new_db_result, 'choice'))

    mh = MetropolisHastings(None, None)
    assert isclose(mh.calc_log_acceptance_ratio("choice", new_db_result.db, db_result.db), acceptance_ratio)
    assert isclose(mh.calc_log_acceptance_ratio("choice", db_result.db, new_db_result.db), -acceptance_ratio)

    #
    # An example with observations in different branches
    #

    def fn():
        if Bernoulli(0.5).sample(name='choice'):
            return Categorical(range(2)).sample(name='rv')
        else:
            Bernoulli(.8).observe(True, name='obs')
            return Categorical(range(3)).sample(name='rv')

    db_result = _db_from_trace(fn, trace=[
        (Bernoulli(0.5), 0),
        (Bernoulli(0.8), True),
        (Categorical(range(3)), 1),
    ])

    new_db_result = _db_from_trace(fn, trace=[
        (Bernoulli(0.5), 1),
        (Categorical(range(2)), 1),
    ])

    acceptance_ratio = db_result.acceptance_ratio(new_db_result, 'choice')
    # Likelihood ratio. Left term is new DB, right term is old DB (including observe probability).
    assert isclose(acceptance_ratio, math.log(1/2) - (math.log(1/3) + math.log(.8)))

    assert isclose(mh.calc_log_acceptance_ratio("choice", new_db_result.db, db_result.db), acceptance_ratio)
    assert isclose(mh.calc_log_acceptance_ratio("choice", db_result.db, new_db_result.db), -acceptance_ratio)


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
