from gorgo import flip, mem, infer, draw_from, factor, condition, \
    Bernoulli, Categorical, Uniform, \
    uniform, recursive_map, recursive_filter, recursive_reduce, \
    map_observe, cps_transform_safe_decorator
from gorgo.interpreter import CPSInterpreter
from gorgo.inference import SimpleEnumeration, LikelihoodWeighting
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
    algebra2 = infer(method="SimpleEnumeration", max_executions=55)(algebra)

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

def test_planning_as_inference():
    import math
    class MDP:
        '''
        This simple task has an always-increasing state, until a terminal
        limit is reached. Rewards are directly based on action indices
        selected.
        '''
        def __init__(self, limit=2):
            self.limit = limit
        def initial_state(self):
            return 0
        def actions(self):
            return [0, 1]
        def is_terminal(self, s):
            return s == self.limit
        def next_state(self, s, a):
            return s + 1
        def reward(self, s, a, ns):
            return -a
    @infer
    def fn(mdp):
        s = mdp.initial_state()
        actions = ()
        while not mdp.is_terminal(s):
            a = Categorical(mdp.actions()).sample()
            actions += (a,)
            ns = mdp.next_state(s, a)
            r = mdp.reward(s, a, ns)
            assert r <= 0
            Bernoulli(math.exp(r)).observe(1)
            s = ns
        return actions
    # Enumerate all action sequences
    s = [(a, b) for a in range(2) for b in range(2)]
    # Total reward for a sequence is exp(-sum(actions))
    expected = Categorical(s, weights=[math.exp(-sum(ev)) for ev in s])
    assert fn(MDP()).isclose(expected)

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

    assert len(SimpleEnumeration(stochastic_mem_func).run().support) == 2**3
    assert len(SimpleEnumeration(no_stochastic_mem_func).run().support) == 2**5


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
    assert SimpleEnumeration(f).run().isclose(Categorical.from_dict({0: 1/3, 1: 1/3, 2: 1/3}))

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
    assert SimpleEnumeration(f).run().isclose(exp)

    def f_with_positive():
        x = flip()
        factor(1.23 if x else -1)
        return x
    exp = {
        True: math.exp(1.23),
        False: math.exp(-1),
    }
    exp = Categorical.from_dict({k: v/sum(exp.values()) for k, v in exp.items()})
    assert SimpleEnumeration(f_with_positive).run().isclose(exp)

def test_condition_statement():
    def f():
        u = uniform()
        condition(.25 < u < .75)
        return u
    samples = LikelihoodWeighting(f, samples=1000).run().support
    assert all(.25 < s < .75 for s in samples)

def test_map_observe():
    @infer
    def f1():
        p = .1 if Bernoulli(.5).sample() else .9
        map_observe(Bernoulli(p), [1, 1, 0, 1])
        return p

    @infer
    def f2():
        p = .1 if Bernoulli(.5).sample() else .9
        [Bernoulli(p).observe(i) for i in [1, 1, 0, 1]]
        return p
    assert f2().isclose(f1())

def test_nested_decorators():
    # module-scope decorator only
    @infer
    def model():
        def f(p):
            return flip(p)
        return f(.2)

    # module-scoped decorator + function-scoped decorator with different return type
    @infer
    def model2a():
        @infer
        def f(p):
            return flip(p)
        return f(.2).sample()
    assert model().isclose(model2a())

    # module-scoped decorator (called) + function-scoped decorator with different return type (called)
    @infer()
    def model2b():
        @infer()
        def f(p):
            return flip(p)
        return f(.2).sample()
    assert model().isclose(model2b())

    # module-scoped decorator + function-scoped decorator with same return type
    @infer
    def model3():
        @mem
        def f(p):
            return flip(p)
        return f(.2)
    assert model().isclose(model3())

def test_sibling_decorators():
    @infer
    def model():
        def f(p):
            return flip(p)
        return f(.2)

    def outer_f(p):
        return flip(p)
    @infer
    def model4():
        return outer_f(.2)
    assert model().isclose(model4())

    @infer
    def outer_f_infer(p):
        return flip(p)
    @infer
    def model5():
        return outer_f_infer(.2).sample()
    assert model().isclose(model5())

    @mem
    def outer_f_mem(p):
        return flip(p)
    @infer
    def model6():
        return outer_f_mem(.2)
    assert model().isclose(model6())

def test_chained_decorators():
    @cps_transform_safe_decorator
    def dec1(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs) + 'a'
        return wrapper

    @cps_transform_safe_decorator
    def dec2(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs) + 'b'
        return wrapper

    @infer
    def model_12():
        @dec2
        @dec1
        def f(p):
            x = '1' if flip(p) else '0'
            return x + "_"
        return f(0.4)

    @infer
    def model_21():
        @dec1
        @dec2
        def f(p):
            x = '1' if flip(p) else '0'
            return x + "_"
        return f(0.4)

    assert model_12().isclose(Categorical.from_dict({'0_ab': 0.6, '1_ab': 0.4}))
    assert model_21().isclose(Categorical.from_dict({'0_ba': 0.6, '1_ba': 0.4}))
