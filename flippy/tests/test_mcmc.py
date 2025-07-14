import math
from flippy import condition
from flippy.distributions.scipy_dists import Bernoulli, Distribution, Normal,  Gamma, Uniform, Beta
from flippy.distributions.builtin_dists import Categorical, Dirichlet
from flippy.inference import SamplePrior, SimpleEnumeration, LikelihoodWeighting, MetropolisHastings
from flippy.tools import isclose
from flippy.interpreter import CPSInterpreter, ReturnState, SampleState, ObserveState
from flippy.inference.mcmc.trace import Trace

from flippy.inference.mcmc.metropolis_hastings import MetropolisHastings as MH

def test_mcmc_trace_and_acceptance_ratio():
    def fn():
        if Bernoulli(0.5).sample(name='choice'):
            return Categorical(range(2)).sample(name='rv')
        else:
            Bernoulli(.8).observe(True, name='obs')
            return Categorical(range(3)).sample(name='rv')

    init_ps = CPSInterpreter().initial_program_state(fn).step()

    # test that trace score computation is correct
    tr = Trace.run_from(
        ps=init_ps,
        sample_site_callback=lambda ps : {
            "choice": 0,
            "rv": 0
        }[ps.name],
        observe_site_callback=lambda ps : ps.value,
        old_trace=None
    )
    assert sum(e.is_sample for e in tr.entries()) == 2
    assert sum(e.score for e in tr.entries() if e.is_sample) == math.log(1/2 * 1/3)
    assert sum(e.score for e in tr.entries() if not e.is_sample) == math.log(.8)

    # test that proposal scores are correct
    mh = MH(
        None,
        None,
        use_drift_kernels=False
    )
    new_tr = Trace.run_from(
        ps=init_ps,
        sample_site_callback=lambda ps : {
            "choice": 0,
            "rv": 0
        }[ps.name],
        observe_site_callback=lambda ps : ps.value,
        old_trace=None
    )
    _, target_site_proposal_score = mh.choose_target_site(
        trace=tr,
        target_site_name='rv',
    )
    _, trace_proposal_score = mh.choose_new_trace(
        old_trace=tr,
        new_trace=new_tr,
        target_site_name='rv',
    )
    assert target_site_proposal_score == math.log(1/2)
    assert trace_proposal_score == math.log(1/3)

    _, trace_proposal_score = mh.choose_new_trace(
        old_trace=tr,
        new_trace=new_tr,
        target_site_name='choice',
    )
    assert trace_proposal_score == math.log(1/2)

    # test acceptance calculation
    log_acceptance_ratio = mh.calc_log_acceptance_ratio(
        new_score=math.log(.5),
        old_score=math.log(.4),
        new_site_score=math.log(.3),
        old_proposal_score=math.log(.2),
        old_site_score=math.log(.15),
        new_proposal_score=math.log(.1),
    )
    isclose(log_acceptance_ratio, (
        math.log((.5*.3*.2)/(.4*.15*.1))
    ))

def test_mcmc_normal_model():
    hyper_mu, hyper_sigma = -1, 1
    obs = [-.75]*10
    sigma = 1
    def normal_model():
        mu = Normal(hyper_mu, hyper_sigma).sample(name='mu')
        Normal(mu, sigma).observe(obs)
        condition(-1.25 < mu < -.5)
        return mu

    seed = 1391299
    mcmc_res = MH(
        function=normal_model,
        samples=2000,
        seed=seed
    ).run()

    lw_res = LikelihoodWeighting(
        function=normal_model,
        samples=2000,
        seed=seed
    ).run()
    assert isclose(mcmc_res.expected_value(), lw_res.expected_value(), atol=.01)

def test_mcmc_gamma_model():
    def gamma_model():
        g = Gamma(3, 2).sample(name='g')
        Uniform(0, g**1.3).observe(0)
        condition(.5 < g < 2)
        return g

    seed = 229932
    mcmc_res = MH(
        function=gamma_model,
        samples=3000,
        burn_in=1000,
        thinning=2,
        seed=seed
    ).run()
    lw_res = LikelihoodWeighting(
        function=gamma_model,
        samples=2000,
        seed=seed
    ).run()
    assert isclose(lw_res.expected_value(), mcmc_res.expected_value(), atol=.01)

def test_mcmc_dirichet_model():
    c1_params = [1, 1, 1]
    c1_data = list('ababababacc')*2
    def model():
        c1 = Dirichlet(c1_params).sample(name='c1')
        dist1 = Categorical(support=list('abc'), probabilities=c1)
        [dist1.observe(d) for d in c1_data]
        return c1

    seed = 13842
    exp_c1 = [n + sum([d == c for d in c1_data]) for c, n in zip('abc', c1_params)]
    exp_c1 = [n / sum(exp_c1) for n in exp_c1]

    mcmc_res = MH(
        function=model,
        samples=2000,
        seed=seed
    ).run()
    for i in [0, 1, 2]:
        est_c1_i = mcmc_res.expected_value(lambda e: e[i])
        assert isclose(est_c1_i, exp_c1[i], atol=.01)

def test_mcmc_beta_model():
    c1_data = [1]*10 + [0]*10
    def model():
        p = Beta(3, 2).sample(name='p')
        bern = Bernoulli(p)
        [bern.observe(d) for d in c1_data]
        return p

    seed = 13842
    exp_p = (3+10) / (3+10 + 2+10)

    mcmc_res = MH(
        function=model,
        samples=1000,
        seed=seed
    ).run()
    est_p = mcmc_res.expected_value()
    assert isclose(est_p, exp_p, atol=.01)


def test_mcmc_categorical_branching_model():
    def model():
        if Bernoulli(0.3).sample(name='choice'):
            return Categorical(range(2)).sample(name='rv')
        else:
            return Categorical(range(3)).sample(name='rv')

    seed = 12949124
    mcmc_res = MH(
        function=model,
        samples=10000,
        seed=seed
    ).run()
    enum_res = SimpleEnumeration(model).run()
    assert isclose(mcmc_res.expected_value(), enum_res.expected_value(), atol=.02)

def test_mcmc_geometric():
    def geometric(p):
        '''
        The probability distribution of the number X of Bernoulli trials needed to get one success.
        https://en.wikipedia.org/wiki/Geometric_distribution
        '''
        x = Bernoulli(p).sample()
        if x == 1:
            return 1
        return 1 + geometric(p)

    param = 0.98
    expected = 1/param

    seed = 13852

    mh_res = MH(
        function=geometric,
        samples=10000,
        seed=seed,
    ).run(param)
    assert isclose(mh_res.expected_value(), expected, atol=.01)

def text_mcmc_categorical_branching_explicit_names():
    def fn():
        if Bernoulli(.5).sample(name="choice"):
            x = Categorical(['a', 'b'], probabilities=[.5, .5]).sample(name='x')
        else:
            x = Categorical(['c', 'b'], probabilities=[.8, .2]).sample(name='x')
        return x
    enum_dist = SimpleEnumeration(fn).run()
    mh_dist = MH(fn, samples=10000, seed=124).run()
    for e in enum_dist:
        assert isclose(enum_dist[e], mh_dist[e], atol=1e-2)

def test_mcmc_initial_value():
    def model():
        p = Beta(3, 2).sample(name='p', initial_value=0.123)
        return p

    mcmc = MetropolisHastings(function=model, samples=10)
    init_ps = mcmc.initial_program_state.step()
    trace = mcmc.generate_initial_trace(init_ps)
    assert trace['p'].value == 0.123
