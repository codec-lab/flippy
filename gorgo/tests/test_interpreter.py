from gorgo import recursive_map, recursive_filter, recursive_reduce, mem
from gorgo.distributions import Bernoulli, Categorical
from gorgo.core import GlobalStore, ReadOnlyProxy
from gorgo.core import SampleState, ReturnState
from gorgo.interpreter import CPSInterpreter
import ast
import pytest
import traceback

from test_transforms import trampoline

def geometric(p):
    x = Bernoulli(p).sample()
    if x == 0:
        return 0
    return 1 + geometric(p)

def algebra():
    def flip():
        return Bernoulli(0.5).sample()
    def NUM():
        return Categorical(range(5)).sample()
    def OP():
        return Categorical(['+', '*']).sample()
    def EQ():
        if flip():
            return NUM()
        else:
            return (NUM(), OP(), EQ())
    return EQ()

def check_trace(func, trace, *, args=(), kwargs={}, return_value):
    ps = CPSInterpreter().initial_program_state(func)
    print(ast.unparse(CPSInterpreter().transform_from_func(func)))
    ps = ps.step(*args, **kwargs)

    for trace_idx, (dist, value) in enumerate(trace):
        assert isinstance(ps, SampleState), (f'{trace_idx=}', ps)
        assert ps.distribution.isclose(dist), (f'{trace_idx=}', ps)
        assert value in dist.support
        ps = ps.step(value)

    assert isinstance(ps, ReturnState)
    assert ps.value == return_value

def test_interpreter():
    check_trace(geometric, [
        (Bernoulli(0.9), 0),
    ], args=(0.9,), return_value=0)

    check_trace(geometric, [
        (Bernoulli(0.8), 1),
        (Bernoulli(0.8), 1),
        (Bernoulli(0.8), 0),
    ], args=(0.8,), return_value=2)

    check_trace(algebra, [
        (Bernoulli(0.5), 1),
        (Categorical(range(5)), 3),
    ], return_value=3)

    check_trace(algebra, [
        (Bernoulli(0.5), 0),
        (Categorical(range(5)), 3),
        (Categorical(['+', '*']), '*'),
        # subtree on right
        (Bernoulli(0.5), 0),
        (Categorical(range(5)), 2),
        (Categorical(['+', '*']), '+'),
        (Bernoulli(0.5), 1),
        (Categorical(range(5)), 4),
    ], return_value=(3, '*', (2, '+', 4)))

def test_cps_map():
    def fn():
        def f(x):
            return Bernoulli(x).sample()
        return sum(recursive_map(f, [.1, .2]))

    check_trace(fn, [
        (Bernoulli(0.1), 1),
        (Bernoulli(0.2), 1),
    ], return_value=2)

def test_cps_filter():
    def fn():
        def f(x):
            return Categorical([1, 2, 3, 4]).sample()
        def is_even(x):
            return x % 2 == 0
        return sum(recursive_filter(is_even, recursive_map(f, [None] * 4)))

    check_trace(fn, [
        (Categorical([1, 2, 3, 4]), 1),
        (Categorical([1, 2, 3, 4]), 2),
        (Categorical([1, 2, 3, 4]), 3),
        (Categorical([1, 2, 3, 4]), 4),
    ], return_value=6)

def test_recursive_reduce():
    def fn():
        return recursive_reduce(lambda acc, x: acc + Bernoulli(x).sample(), [.1, .2], 0)

    check_trace(fn, [
        (Bernoulli(0.1), 1),
        (Bernoulli(0.2), 1),
    ], return_value=2)

def test_list_comprehension():
    # Simple case
    expected = [0, 1, 4]
    def fn():
        return [x**2 for x in range(3)]
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Multiple if statements.
    expected = [0, 2]
    def fn():
        return [x for x in range(5) if x % 2 == 0 if x < 3]
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Multiple generators
    expected = [(0, 0), (0, 1), (0, 2), (2, 0)]
    def fn():
        return [
            (x, y)
            for x in range(4)
            if x % 2 == 0
            for y in range(5)
            if x + y < 3
        ]
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Nested comprehensions
    expected = [[0], [0, 1], [0, 1, 2]]
    def fn():
        return [
            [y for y in range(x+1)]
            for x in range(3)
        ]
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Set comprehensions
    expected = {0, 1, 4, 9}
    def fn():
        return {
            x**2
            for x in [-3, -2, -1, 0, 1, 2, 3]
        }
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Dict comprehensions
    expected = {0: 0, 1: 1, 2: 4, 3: 9}
    def fn():
        return {
            x: x**2
            for x in range(4)
        }
    assert fn() == expected
    check_trace(fn, [], return_value=expected)

    # Checking something stochastic.
    def fn():
        return sum([Bernoulli(x).sample() for x in [.1, .2, .3]])
    check_trace(fn, [
        (Bernoulli(0.1), 1),
        (Bernoulli(0.2), 0),
        (Bernoulli(0.3), 1),
    ], return_value=2)

def test_check_exception():
    def test_fn():
        raise Exception('expected exception')
        return 3
    with pytest.raises(Exception) as e:
        check_trace(test_fn, [], return_value=3)
    # The right exception was raised.
    assert 'expected exception' in str(e)

    # Now check exception content to make sure it references code.
    _, exc, tb = e._excinfo
    # We show compiled code, which in this case isn't particularly readable.
    # This line will have to change depending on our compiled output.
    exception_line = 'raise __v0'

    # First, we just check that the traceback will render the code.
    formatted = ''.join(traceback.format_exception(exc, None, tb))
    assert exception_line in formatted

    # This is a more detailed check of the contents, ensuring the filename is correct in traceback.
    last_entry = traceback.extract_tb(tb)[-1]
    assert 'test_fn' in last_entry.filename
    assert hex(id(test_fn)).removeprefix('0x') in last_entry.filename
    assert last_entry.line == exception_line

def test_control_flow_or():
    def fn_or():
        return Bernoulli(0.5).sample() or Bernoulli(0.5).sample()

    check_trace(fn_or, [
        (Bernoulli(0.5), 0),
        (Bernoulli(0.5), 1),
    ], return_value=1)

    check_trace(fn_or, [
        (Bernoulli(0.5), 1),
    ], return_value=1)

def test_control_flow_and():
    def fn_and():
        return Bernoulli(0.5).sample() and Bernoulli(0.5).sample()

    check_trace(fn_and, [
        (Bernoulli(0.5), 1),
        (Bernoulli(0.5), 1),
    ], return_value=1)

    check_trace(fn_and, [
        (Bernoulli(0.5), 0),
    ], return_value=0)

def test_conditional_reassign():
    def fn():
        x = 0
        if Bernoulli(0.5).sample():
            x = 42
        return x

    check_trace(fn, [(Bernoulli(0.5), 0)], return_value=0)
    check_trace(fn, [(Bernoulli(0.5), 1)], return_value=42)

def test_load_then_store_in_new_scope():
    def fn():
        x = 0
        z = Bernoulli(0.5).sample()
        x = x + z
        return x

    check_trace(fn, [(Bernoulli(0.5), 0)], return_value=0)
    check_trace(fn, [(Bernoulli(0.5), 1)], return_value=1)

def test_conditionally_defined():
    # This can break implementations of scope passing that assume
    # variable lists can be fully statically determined.
    def fn():
        z = Bernoulli(0.5).sample()

        if z:
            y = 42

        if z:
            return y
        else:
            return -1

    check_trace(fn, [(Bernoulli(0.5), 0)], return_value=-1)
    check_trace(fn, [(Bernoulli(0.5), 1)], return_value=42)

def test_closure_issues():
    # Making sure we throw for important closure issues.

    # If incorrect, this returns 'bad'.
    with pytest.raises(SyntaxError) as err:
        def fn():
            def nested():
                return x
            x = 'bad'
            Bernoulli(0.5).sample() # Arbitrary. To put rest of function in continuation.
            x = 'good'
            return nested()
        check_trace(fn, [(Bernoulli(0.5), 0)], return_value='good')
    assert 'must be immutable' in str(err)

    # If incorrect, this fails to run with NameError.
    with pytest.raises(SyntaxError) as err:
        def fn():
            def nested():
                return x
            Bernoulli(0.5).sample() # Arbitrary. To put rest of function in continuation.
            x = 'good'
            return nested()
        check_trace(fn, [(Bernoulli(0.5), 0)], return_value='good')
    assert 'must be defined before' in str(err)

def test_global_store_proxy():
    global_store = None
    def f():
        return global_store.get('a', 'no value')

    cps = CPSInterpreter()
    code = cps.transform_from_func(f)
    context = {"_cps": cps, 'global_store': cps.global_store_proxy}
    exec(ast.unparse(code), context)
    f = context['f']

    with cps.set_global_store(GlobalStore()):
        assert trampoline(f()) == 'no value'

    with cps.set_global_store(GlobalStore({'a': 100})):
        assert trampoline(f()) == 100

proxy_forking_bags = Categorical(['bag0', 'bag1', 'bag2'])
def proxy_forking_value(_bag):
    return Categorical(range(10)).sample()
def proxy_forking():
    value = mem(proxy_forking_value)
    return [
        value(proxy_forking_bags.sample())
        for _ in range(5)
    ]

def test_global_store_proxy_forking():
    cps = CPSInterpreter()
    s0 = cps.initial_program_state(proxy_forking).step()
    assert s0.init_global_store.store == {}

    s1a = s0.step('bag0').step(3)
    assert s0.init_global_store.store == {}, 'Make sure original state was not modified'
    assert s1a.init_global_store.store == {(proxy_forking_value, ('bag0',), ()): 3}

    # A sanity check, we shouldn't need to resample for bag0 now.
    s = s1a.step('bag0')
    assert isinstance(s, SampleState) and s.distribution == proxy_forking_bags

    # Now, we test forking by restarting at s0
    s1b = s0.step('bag1').step(7)
    assert s0.init_global_store.store == {}, 'Make sure original state was not modified'
    assert s1a.init_global_store.store == {(proxy_forking_value, ('bag0',), ()): 3}, 'Make sure sibling state was not modified'
    assert s1b.init_global_store.store == {(proxy_forking_value, ('bag1',), ()): 7}

    # This forked state should not pick up on state from s1a
    s2b = s1b.step('bag0').step(5)
    assert s0.init_global_store.store == {}, 'Make sure original state was not modified'
    assert s1a.init_global_store.store == {(proxy_forking_value, ('bag0',), ()): 3}, 'Make sure sibling state was not modified'
    assert s1b.init_global_store.store == {(proxy_forking_value, ('bag1',), ()): 7}, 'Make sure original state was not modified'
    assert s2b.init_global_store.store == {(proxy_forking_value, ('bag1',), ()): 7, (proxy_forking_value, ('bag0',), ()): 5}, 'Make sure original state was not modified'
