from gorgo import flip, mem, infer, draw_from, factor, condition, \
    Bernoulli, Categorical, Uniform, \
    uniform, recursive_map, recursive_filter, recursive_reduce
from gorgo.interpreter import CPSInterpreter
from gorgo.inference import Enumeration, LikelihoodWeighting
from gorgo.core import ReturnState
import math


def algebra():
    def flip():
        return Bernoulli(0.5).sample()
    def NUM():
        return Categorical(range(5)).sample()
    def OP():
        if flip():
            return '+'
        else:
            return '*'
    def EQ():
        if flip():
            return NUM()
        else:
            return (NUM(), OP(), EQ())
    return EQ()

def test_algebra():
    algebra2 = infer(max_executions=55)(algebra)

    results = algebra2()

    num_ct = 0
    expr_ct = 0
    for e in results.support:
        if isinstance(e, int):
            num_ct += 1
        else:
            assert (
                isinstance(e, tuple) and
                len(e) == 3 and
                isinstance(e[0], int) and
                e[1] in ('*', '+') and
                isinstance(e[2], int)
            )
            expr_ct += 1
    assert num_ct == 5
    assert expr_ct == 5 * 2 * 5

def world_prior():
    return Categorical(range(4)).sample()

def utterance_prior():
    return Categorical([
        'some of the people are nice',
        'all of the people are nice',
        'none of the people are nice',
    ]).sample()

def meaning(utt, world):
    return (
        world > 0 if utt == 'some of the people are nice' else
        world == 3 if utt == 'all of the people are nice' else
        world == 0 if utt == 'none of the people are nice' else
        True
    )

@infer
def literal_listener(utterance):
    world = world_prior()
    m = meaning(utterance, world)
    Bernoulli(1.0).observe(m)
    return world

@infer
def speaker(world) -> str:
    utterance = utterance_prior()
    L = literal_listener(utterance)
    L.observe(world)
    return utterance

@infer
def listener(utterance):
    world = world_prior()
    S = speaker(world)
    S.observe(utterance)
    return world

def test_scalar_implicature():
    assert listener('some of the people are nice').isclose(Categorical(
        [1, 2, 3],
        probabilities=[4/9, 4/9, 1/9],
    ))

def test_infer():
    def f0():
        return Bernoulli(.4).sample()
    f0 = infer(f0)
    @infer
    def f1():
        return Bernoulli(.4).sample()
    @infer()
    def f2():
        return Bernoulli(.4).sample()
    @infer(cache_size=10)
    def f3():
        return Bernoulli(.4).sample()
    assert f1().isclose(f2())
    assert f1().isclose(f0())
    assert f1().isclose(f3())


def test_builtins():
    @infer
    def model():
        return abs(Categorical([-1, 0, 1]).sample())
    assert model().isclose(Categorical(
        [0, 1],
        probabilities=[1/3, 2/3],
    ))

    @infer
    def model():
        mydict = {}
        return tuple(mydict.items())
    assert model().isclose(Categorical([(),]))

def test_cps_map():
    assert recursive_map(lambda x: x ** 2, []) == []
    assert recursive_map(lambda x: x ** 2, list(range(5))) == [0, 1, 4, 9, 16]

def test_cps_filter():
    assert recursive_filter(lambda x: x % 2 == 0, []) == []
    assert recursive_filter(lambda x: x % 2 == 0, list(range(5))) == [0, 2, 4]

def test_recursive_reduce():
    sumfn = lambda acc, el: acc + el

    assert recursive_reduce(sumfn, [], 0) == 0
    assert recursive_reduce(sumfn, [1], 0) == 1
    assert recursive_reduce(sumfn, [1, 2], 0) == 3
    assert recursive_reduce(sumfn, [1, 2, 3], 0) == 6

    assert recursive_reduce(sumfn, [[3, 4], [5]], []) == [3, 4, 5]

def test_stochastic_memoization():
    def stochastic_mem_func():
        def g(i):
            return flip()
        g = mem(g)
        return (g(1), g(1), g(2), g(2), flip())

    def no_stochastic_mem_func():
        def g(i):
            return flip()
        return (g(1), g(1), g(2), g(2), flip())

    assert len(Enumeration(stochastic_mem_func).run().support) == 2**3
    assert len(Enumeration(no_stochastic_mem_func).run().support) == 2**5


def test_mem_basic():
    def with_mem():
        def f(i):
            return Uniform(0, 1).sample()
        f = mem(f)
        x = f(0)
        return (x + x, f(0) + f(0))

    u = .456
    ps = CPSInterpreter().initial_program_state(with_mem)
    ps = ps.step().step(u)
    assert isinstance(ps, ReturnState)
    assert ps.value == (u*2, u*2)

    def without_mem():
        def f(i):
            return Uniform(0, 1).sample()
        x = f(0)
        return (x + x, f(0) + f(0))

    ps = CPSInterpreter().initial_program_state(without_mem)
    ps = ps.step().step(u).step(0).step(0)
    assert isinstance(ps, ReturnState)
    assert ps.value == (u*2, 0)


def test_draw_from():
    def f():
        return draw_from(3)
    assert Enumeration(f).run().isclose(Categorical.from_dict({0: 1/3, 1: 1/3, 2: 1/3}))

def test_factor_statement():
    def f():
        x = flip(.4)
        y = flip(.7)
        factor(math.log(.2) if x == y else math.log(.8))
        return (x, y)

    exp = {
        (0, 0): .2*.6*.3,
        (0, 1): .8*.6*.7,
        (1, 0): .8*.4*.3,
        (1, 1): .2*.4*.7,
    }
    exp = Categorical.from_dict({k: v/sum(exp.values()) for k, v in exp.items()})
    assert Enumeration(f).run().isclose(exp)

    def f_with_positive():
        x = flip()
        factor(1.23 if x else -1)
        return x
    exp = {
        True: math.exp(1.23),
        False: math.exp(-1),
    }
    exp = Categorical.from_dict({k: v/sum(exp.values()) for k, v in exp.items()})
    assert Enumeration(f_with_positive).run().isclose(exp)

def test_condition_statement():
    def f():
        u = uniform()
        condition(.25 < u < .75)
        return u
    samples = LikelihoodWeighting(f, samples=1000).run().support
    assert all(.25 < s < .75 for s in samples)
