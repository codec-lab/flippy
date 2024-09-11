import collections
import math

from frozendict import frozendict
from gorgo import flip, mem, condition, draw_from
from gorgo.distributions import Bernoulli, Distribution, Categorical, Dirichlet, Normal, Gamma, Uniform
from gorgo.inference import SamplePrior, SimpleEnumeration, LikelihoodWeighting, _distribution_from_inference
from gorgo.inference.enumeration import Enumeration
from gorgo.tools import isclose
from gorgo.interpreter import CPSInterpreter, ReturnState, SampleState, ObserveState
from gorgo.callentryexit import register_call_entryexit
from gorgo.map import independent_map

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

def test_Enumeration_binom():
    def binom(k, p):
        ct = 0
        for i in range(k):
            ct += Bernoulli(p).sample()
        return ct

    enum = Enumeration(binom, _call_cache_size=100, _emit_call_entryexit=True)
    enum.run(2, .5)
    assert enum._call_cache.hits == 0
    assert enum._call_cache.misses == 7
    assert len(enum._call_cache) == 7

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
        (1, 1): 2,
        (1, 2): 1,
    }
    assert collections.Counter(simpler_keys) == binom2keys

    enum = Enumeration(binom, _call_cache_size=100, _emit_call_entryexit=True)
    enum.run(3, .5)
    assert enum._call_cache.hits == 2
    assert enum._call_cache.misses == 13
    assert len(enum._call_cache) == 13

    simpler_keys = [
        (scope.get('i'), scope.get('ct'))
        for (_, (scope,), _, _) in enum._call_cache.keys()
    ]
    assert len(set(simpler_keys)) == 10
    assert collections.Counter(simpler_keys) == {
        **binom2keys,
        (2, 0): 1,
        (2, 1): 2,
        (2, 2): 2,
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

def test_enumerating_loops():
    # HACK Goal here was for bad naming with continue to break things, but this works
    # TODO make test that breaks with continue naming issues
    def loop_continue_fn():
        ct = 0
        i = 0
        while i < 3:
            if Bernoulli(.5).sample():
                ct += i * 2
                i += 1
                continue
            ct += i
            i += 1
        return ct

    p = .25
    rv1 = SimpleEnumeration(loop_continue_fn).run(p)
    rv2 = Enumeration(loop_continue_fn).run(p)
    assert rv1.isclose(rv2)
