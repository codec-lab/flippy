import math
from gorgo import condition
from gorgo.distributions import Bernoulli, Distribution, Categorical, Dirichlet, Normal,  Gamma, Uniform
from gorgo.inference import SamplePrior, Enumeration, LikelihoodWeighting, MetropolisHastings
from gorgo.inference.metropolis_hastings import Entry
from gorgo.tools import isclose
from gorgo.interpreter import CPSInterpreter, ReturnState, SampleState, ObserveState
from gorgo.inference.metropolis_hastings import Mapping, Hashable
import dataclasses

from gorgo.inference.mcmc.prior_proposal import PriorProposalMCMC

def test_mcmc_normal_model():
    hyper_mu, hyper_sigma = -1, 1
    obs = [-.75]*10
    sigma = 1
    def normal_model():
        mu = Normal(hyper_mu, hyper_sigma).sample(name='mu')
        Normal(mu, sigma).observe(obs)
        condition(-1.25 < mu < -.5)
        return mu

    seed = 2191299
    mcmc_res = PriorProposalMCMC(
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
        Uniform(0, g**2).observe(0)
        return g

    seed = 12999124
    mcmc_res = PriorProposalMCMC(
        function=gamma_model,
        samples=5000,
        seed=seed
    ).run()
    lw_res = LikelihoodWeighting(
        function=gamma_model,
        samples=5000,
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

    mcmc_res = PriorProposalMCMC(
        function=model,
        samples=5000,
        seed=seed
    ).run()
    for i in [0, 1, 2]:
        est_c1_i = mcmc_res.expected_value(lambda e: e[i])
        assert isclose(est_c1_i, exp_c1[i], atol=.01)

def test_mcmc_categorical_branching_model():
    def model():
        if Bernoulli(0.3).sample(name='choice'):
            return Categorical(range(2)).sample(name='rv')
        else:
            return Categorical(range(3)).sample(name='rv')

    seed = 12949124
    mcmc_res = PriorProposalMCMC(
        function=model,
        samples=50000,
        seed=seed
    ).run()
    enum_res = Enumeration(model).run()
    assert isclose(mcmc_res.expected_value(), enum_res.expected_value(), atol=.01)
