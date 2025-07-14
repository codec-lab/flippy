import collections
import math
import pytest

from frozendict import frozendict
from flippy import flip, mem, condition, draw_from
from flippy.distributions.builtin_dists import \
    Bernoulli, Distribution, Categorical, Gaussian, Uniform, Binomial, Poisson
from flippy.inference import SamplePrior, SimpleEnumeration, LikelihoodWeighting, _distribution_from_inference
from flippy.inference.enumeration import Enumeration
from flippy.inference.max_marg_post import MaximumMarginalAPosteriori
from flippy.tools import isclose
from flippy.interpreter import CPSInterpreter, ReturnState, SampleState, ObserveState
from flippy.callentryexit import register_call_entryexit
from flippy.map import independent_map

def geometric(p):
    '''
    The probability distribution of the number X of Bernoulli trials needed to get one success.
    https://en.wikipedia.org/wiki/Geometric_distribution
    '''
    x = Bernoulli(p).sample()
    if x == 1:
        return 1
    return 1 + geometric(p)

def geometric_iter(p):
    ct = 1
    while Bernoulli(p).sample() == 0:
        ct += 1
    return ct

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
    for fn in [geometric, geometric_iter]:
        rv = SimpleEnumeration(fn, max_executions=100).run(param)
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

        dist = _distribution_from_inference(SimpleEnumeration(model).run())
        print('Enumeration', dist)
        assert dist.isclose(expected_dist)

        dist = _distribution_from_inference(LikelihoodWeighting(model, samples=samples, seed=seed).run())
        print('LikelihoodWeighting', dist)
        assert dist.isclose(expected_dist, atol=1e-1)

def test_graph_enumeration():
    def f1():
        def g():
            return flip(.4) + flip(.7) + flip(.9) + flip(.2) + flip(.51)
        return g() + g()

    def f2():
        def g(i):
            return flip() + flip()
        g = mem(g)
        i = flip()
        j = flip()
        return g(i) + g(j)

    def f3():
        i = flip(.3)
        j = flip(.72)
        condition(.9 if i + j == 1 else .3)
        return i + j

    def f4():
        # @register_call_entryexit
        def g(i):
            return flip(.61, name='a') + flip(.77, name='b')
        x = flip(.3, name='x')
        return x + g(1)

    def f5():
        # @register_call_entryexit
        def g(i):
            Bernoulli(.3).observe(i)
            return flip(.61, name='a') + flip(.77, name='b')
        x = flip(.3, name='x')
        return x + g(x)

    def f6():
        num = lambda : draw_from(range(2))
        op = lambda : '+' if flip(.5) else '*'
        def eq(d):
            if d == 0 or flip(.34):
                return num()
            else:
                return (num(), op(), eq(d - 1))
        return eq(3)

    def f7():
        return flip(0)

    def f8():
        def g(i):
            if i < 0.5:
                return 1
            else:
                return flip(i)
        x = independent_map(g, (.1, .2, .3, .4, .5, .6, .7))
        return x

    def f9():
        # @register_call_entryexit
        def g(i):
            return flip(i)
        x = g(.2)
        condition(1)
        return x

    def f10():
        def g(i):
            return flip(i)
        x = independent_map(g, (.2,))
        condition(1)
        return x

    def f11():
        # @register_call_entryexit
        def f(a, b):
            condition(a == b)
        a = flip()
        b = flip()
        f(a, b)
        return a + b

    test_models = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]

    failed = {}
    for f in test_models:
        e_res = SimpleEnumeration(f).run()
        ge_res = Enumeration(f).run()
        if not e_res.isclose(ge_res):
            failed[f] = (e_res, ge_res)

    if failed:
        raise AssertionError('\n'.join([
            "Failed models:",
            *[f"{f.__name__}:\n\t{e_res}\n\t{ge_res}" for f, (e_res, ge_res) in failed.items()]
        ]))

def test_hashing_program_states_with_list_and_dict():
    def f():
        a = []
        b = {}
        return flip()

    ps = CPSInterpreter().initial_program_state(f)
    assert id(ps.step()) != id(ps.step())
    assert hash(ps.step()) == hash(ps.step())

def test_graph_enumeration_callsite_caching():
    def model():
        @register_call_entryexit
        def f(p):
            return flip(p) + flip(p)
        x = f(.3) + f(.3) + f(.3)
        return x

    e_res = SimpleEnumeration(model).run()
    ge_res_nocache = Enumeration(model, _call_cache_size=0, _emit_call_entryexit=False).run()
    ge_res_cache = Enumeration(model, _call_cache_size=1000, _emit_call_entryexit=False).run()
    assert e_res.isclose(ge_res_nocache)
    assert e_res.isclose(ge_res_cache)

def test_graph_enumeration_callsite_caching_with_mem():
    def model():
        @mem
        def g(i):
            return .3 if flip(.6) else .63
        @register_call_entryexit
        def f():
            p = g(0)
            return flip(p) + flip(p)
        x = f() + f() + f()
        return x

    e_res = SimpleEnumeration(model).run()
    ge_res_nocache = Enumeration(model, _call_cache_size=0, _emit_call_entryexit=False).run()
    ge_res_cache = Enumeration(model, _call_cache_size=1000, _emit_call_entryexit=False).run()
    assert e_res.isclose(ge_res_nocache)
    assert e_res.isclose(ge_res_cache)

def test_graph_enumeration_callsite_caching_lru_cache():
    def model():
        @register_call_entryexit
        def f(p):
            return flip(p)
        x = f(.1) + f(.2) + f(.3) + f(.4)
        return x

    ge = Enumeration(model, _call_cache_size=2, _emit_call_entryexit=False)
    e_res = SimpleEnumeration(model).run()
    ge_res_cache = ge.run()
    assert e_res.isclose(ge_res_cache)
    assert len(ge._call_cache) == 2
    assert [args[0] for _, args, _, _ in ge._call_cache.keys()] == [.3, .4]

def test_Enumeration_call_cache_outside_function():
    # after m(1):
    #     misses = {f(-1), f(-2)}     n = 2
    #     hits = [f(-1)]    n = 1
    # after m(2):
    #     misses = {f(-1), f(-2), f(-3)}   n = 3
    #     hits = [f(-1), f(-2), f(-2)]   n = 3
    def f(i):
        return 0
    def m(i):
        return f(-i) + f(-i) + f(-(i + 1))

    enum = Enumeration(m, _call_cache_size=10, _emit_call_entryexit=True)
    enum.run(1)
    assert enum._call_cache.hits == 1
    assert enum._call_cache.misses == 2
    assert len(enum._call_cache) == 2
    enum.run(2)
    assert enum._call_cache.hits == 3
    assert enum._call_cache.misses == 3
    assert len(enum._call_cache) == 3

def test_Enumeration_call_cache_nested_function():
    # after m(1):
    #     misses = {f(-1), f(-2)}     n = 2
    #     hits = [f(-1)]    n = 1
    # after m(2):
    #     misses = {f(-1), f(-2), f(-3)}   n = 3
    #     hits = [f(-1), f(-2), f(-2)]   n = 3
    def m(i):
        def f(i):
            return 0
        return f(-i) + f(-i) + f(-(i + 1))

    enum = Enumeration(m, _call_cache_size=10, _emit_call_entryexit=True)
    enum.run(1)
    assert enum._call_cache.hits == 1
    assert enum._call_cache.misses == 2
    assert len(enum._call_cache) == 2
    enum.run(2)
    assert enum._call_cache.hits == 3
    assert enum._call_cache.misses == 3
    assert len(enum._call_cache) == 3

    def m(i):
        def f(i):
            return i + 1
        return f(f(f(i)))

    enum = Enumeration(m, _call_cache_size=10, _emit_call_entryexit=True)
    enum.run(1)
    assert enum._call_cache.hits == 0
    assert enum._call_cache.misses == 3
    assert len(enum._call_cache) == 3

def test_Enumeration_binom():
    def binom(k, p):
        ct = 0
        for i in range(k):
            ct += Bernoulli(p).sample()
        return ct

    enum = Enumeration(binom, _call_cache_size=100, _emit_call_entryexit=True)
    enum.run(2, .5)
    assert enum._call_cache.hits == 1
    assert enum._call_cache.misses == 6
    assert len(enum._call_cache) == 6

    simpler_keys = [
        (scope.get('i'), scope.get('ct'))
        for (_, (scope,), _, _) in enum._call_cache.keys()
    ]
    assert len(set(simpler_keys)) == 6
    binom2keys = {
        (None, 0): 1,
        (0, 0): 1,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 1,
        (1, 2): 1,
    }
    assert collections.Counter(simpler_keys) == binom2keys

    enum = Enumeration(binom, _call_cache_size=100, _emit_call_entryexit=True)
    enum.run(3, .5)
    assert enum._call_cache.hits == 3
    assert enum._call_cache.misses == 10
    assert len(enum._call_cache) == 10

    simpler_keys = [
        (scope.get('i'), scope.get('ct'))
        for (_, (scope,), _, _) in enum._call_cache.keys()
    ]
    assert len(set(simpler_keys)) == 10
    assert collections.Counter(simpler_keys) == {
        **binom2keys,
        (2, 0): 1,
        (2, 1): 1,
        (2, 2): 1,
        (2, 3): 1,
    }

def test_enumerating_class_method():
    def flip():
        p = .66
        return Bernoulli(p).sample()

    class C:
        p = .66
        def flip(self):
            return Bernoulli(self.p).sample()
    c = C()
    method_res = Enumeration(C.flip).run(c)
    func_res = Enumeration(flip).run()
    assert method_res.isclose(func_res)

def test_Enumeration__set_cont_var_func():
    def f():
        return Uniform().sample()

    def model():
        p = f()
        return p

    continous_handler1 = lambda ps: [0.5]
    continous_handler2 = lambda ps: [0.25]

    enum = Enumeration(
        model,
        cont_var_func=None,
        _emit_call_entryexit=True
    )

    with pytest.raises(AssertionError) as e:
        enum.run()
    assert "Continuous sample state values must be provided for continuous distributions" in str(e.value)

    enum.set_cont_var_func(continous_handler1)
    res1 = enum.run()
    assert res1.isclose(Categorical.from_dict({0.5: 1.0}))

    enum.set_cont_var_func(continous_handler2)
    res2 = enum.run()
    assert res2.isclose(Categorical.from_dict({0.25: 1.0}))

def test_Enumeration__max_states_with_recursive_model():
    def deep_recurse(d):
        if d == 0:
            return 0
        i = Bernoulli(.9).sample(name=f'i{d}') #asymmetric otherwise its just bfs
        return i + deep_recurse(d - 1)

    n_sample_states_visited = [0]
    def cb(ps):
        if isinstance(ps, SampleState):
            n_sample_states_visited[0] += 1
        return

    enum_limit = Enumeration(
        deep_recurse,
        max_states=10,
        _emit_call_entryexit=False,
        _state_visit_callback=cb
    )
    enum_limit_res = enum_limit.run(5)
    assert n_sample_states_visited[0] == 11

    enum_full_res = Enumeration(deep_recurse).run(5)
    assert not enum_full_res.isclose(enum_limit_res)

def test_Enumeration__set_cont_var_func__with_without_entry_exit():
    def f():
        return Uniform().sample()

    def model():
        return f() + f()

    cont_func_no_entryexit_visits = [0]
    def cont_func_no_entryexit(ps: SampleState):
        cont_func_no_entryexit_visits[0] += 1
        return [.1]

    Enumeration(
        model,
        _emit_call_entryexit=False,
        cont_var_func=cont_func_no_entryexit
    ).run()

    cont_func_with_entryexit_visits = [0]
    def cont_func_with_entryexit(ps: SampleState):
        cont_func_with_entryexit_visits[0] += 1
        return [.1]

    Enumeration(
        model,
        _emit_call_entryexit=True,
        cont_var_func=cont_func_with_entryexit
    ).run()

    # both continuous variable functions should have been accessed twice since
    # they should not be cached
    assert cont_func_no_entryexit_visits[0] == 2
    assert cont_func_with_entryexit_visits[0] == 2

def test_Enumeration__set_cont_var_func__with_without_entry_exit_nested():
    # we should always be calling into g(None) and then f()
    def f():
        return Uniform().sample()
    def g(p=None):
        if p is None:
            p = f()
        return p

    def model():
        return g() + g(.3) + g(.3) + g() + g(.3) + g()

    cont_var_func_visits = [0]
    def cont_var_func(ps: SampleState):
        cont_var_func_visits[0] += 1
        return [.1]
    enum = Enumeration(model, cont_var_func=cont_var_func, _emit_call_entryexit=True)
    dist = enum.run()
    assert dist.isclose(Categorical([.1 + .3 + .3 + .1 + .3 + .1]))
    assert cont_var_func_visits[0] == 3
    assert len(enum._call_cache) == 1
    assert enum._call_cache.hits == 2
    assert enum._call_cache.misses == 2 + 1 + 0 + 2 + 0 + 2

def test_Enumeration__cont_var_func__pre_continuous_branching():
    def f(p=None):
        if p is None:
            p = Uniform().sample()
        return p

    def model():
        _ = Bernoulli(.3).sample()
        return f(.73) + f(.73) + f() + f()

    cont_var_func_visits = [0]
    def cont_var_func(ps):
        cont_var_func_visits[0] += 1
        return [.4]

    enum = Enumeration(model, cont_var_func=cont_var_func)
    dist = enum.run()
    assert dist.isclose(Categorical([.73 + .73 + .4 + .4]))
    assert cont_var_func_visits[0] == (0 + 0 + 1 + 1) + (0 + 0 + 1 + 1)
    assert len(enum._call_cache) == 1
    assert enum._call_cache.hits == (0 + 1 + 0 + 0) + (1 + 1 + 0 + 0)
    assert enum._call_cache.misses == (1 + 0 + 1 + 1) + (0 + 0 + 1 + 1)

def test_Enumeration__cont_var_func__post_continuous_branching():
    def f(p=None):
        if p is None:
            p = Uniform().sample()
        return p

    def model():
        res = f(.73) + f(.73) + f() + f()
        _ = Bernoulli(.3).sample()
        return res

    cont_var_func_visits = [0]
    def cont_var_func(ps):
        cont_var_func_visits[0] += 1
        return [.4]

    enum = Enumeration(model, cont_var_func=cont_var_func)
    dist = enum.run()
    assert dist.isclose(Categorical([.73 + .73 + .4 + .4]))
    assert cont_var_func_visits[0] == (0 + 0 + 1 + 1)
    assert len(enum._call_cache) == 1
    assert enum._call_cache.hits == (0 + 1 + 0 + 0)
    assert enum._call_cache.misses == (1 + 0 + 1 + 1)

def test_MaximumMarginalAPosteriori__fit_binomial():
    def model1():
        p = Uniform().sample()
        x = Binomial(9, p).sample()
        condition(x == 6)
        return p

    dist = MaximumMarginalAPosteriori(model1).run()
    assert len(dist) == 1
    assert isclose(dist.sample(), 6/9, atol=1e-4)

    def model2():
        p1 = Uniform().sample()
        p2 = Uniform().sample()
        p = p1 + p2
        condition(p < 1)
        Binomial(9, p).observe(6)
        return p

    dist = MaximumMarginalAPosteriori(model2, seed=42).run()
    assert len(dist) == 1
    assert isclose(dist.sample(), 6/9, atol=1e-4)

    def model3():
        p1 = Uniform().sample()
        p2 = Uniform().sample()
        assert p1 != p2
        p = p1 if Bernoulli(.5).sample() else p2*.9
        Binomial(9, p).observe(6)
        return p

    dist = MaximumMarginalAPosteriori(model3).run()
    assert isclose(dist.expected_value(), 6/9, atol=1e-4)

def test_MaximumMarginalAPosteriori__fit_gaussian():
    def m1(data, prior_mu=0, prior_sig=1, obs_sig=.1):
        mu = Gaussian(prior_mu, prior_sig).sample()
        [Gaussian(mu, obs_sig).observe(d) for d in data]
        return mu

    # conjugate update
    def m1_conj(data, prior_mu=0, prior_sig=1, obs_sig=.1):
        post_sig = 1/((1/prior_sig**2) + (len(data)/(obs_sig**2)))
        post_mu = post_sig*(prior_mu/prior_sig**2 + sum(data)/(obs_sig**2))
        return post_mu

    data = (1.579, .667, .234)
    # Maximum A Posteriori
    dist = MaximumMarginalAPosteriori(m1, maximum_likelihood=False).run(data)
    assert len(dist) == 1
    assert isclose(dist.expected_value(), m1_conj(data), atol=1e-4)
    map_mu = dist.expected_value()
    exp_score = sum(Gaussian(map_mu, .1).log_probability(d) for d in data) + \
        Gaussian(0, 1).log_probability(map_mu)
    est_score = np.log(dist.marginal_likelihood)
    assert isclose(est_score, exp_score), (est_score, exp_score)

    # Maximum Likelihood
    dist_ml = MaximumMarginalAPosteriori(m1, maximum_likelihood=True).run(data)
    assert len(dist_ml) == 1
    assert isclose(dist_ml.expected_value(), sum(data)/len(data), atol=1e-4)
    ml_mu = dist_ml.expected_value()
    exp_score = sum(Gaussian(ml_mu, .1).log_probability(d) for d in data)
    est_score = np.log(dist_ml.marginal_likelihood)
    assert isclose(est_score, exp_score), (est_score, exp_score)

def test_MaximumMarginalAPosteriori__fit_unbounded_number_of_vars():
    def f(i=0):
        p = Uniform().sample(name=f"p{i}")
        if p < .4:
            return i, p
        return f(i=i + 1)

    def m():
        i, p = f()
        Poisson(5).observe(i)
        Binomial(10, p).observe(5)
        return i, p

    # Nelder-Mead works here; Powell tends to get stuck or hang
    dist = MaximumMarginalAPosteriori(m, method="Nelder-Mead").run()
    assert isclose(dist.expected_value(lambda x: x[1]), .4, atol=1e-4)

