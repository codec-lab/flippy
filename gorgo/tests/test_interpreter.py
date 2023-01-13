from gorgo import cps_map, cps_filter, cps_reduce
from gorgo.core import Bernoulli, Categorical
from gorgo.core import SampleState, ReturnState
from gorgo.interpreter import CPSInterpreter, ParsingError
import ast
import pytest
import traceback

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
        return sum(cps_map(f, [.1, .2]))

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
        return sum(cps_filter(is_even, cps_map(f, [None] * 4)))

    check_trace(fn, [
        (Categorical([1, 2, 3, 4]), 1),
        (Categorical([1, 2, 3, 4]), 2),
        (Categorical([1, 2, 3, 4]), 3),
        (Categorical([1, 2, 3, 4]), 4),
    ], return_value=6)

def test_cps_reduce():
    def fn():
        return cps_reduce(lambda acc, x: acc + Bernoulli(x).sample(), [.1, .2], 0)

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

def test_interpreting_lambda_as_model():
    model = lambda : 10
    assert CPSInterpreter().interpret(model)(_cont=lambda v: v)() == 10

    model = (lambda : lambda : 123)()
    assert CPSInterpreter().interpret(model)(_cont=lambda v: v)() == 123

    model = (lambda p : lambda : 123)(1)
    assert CPSInterpreter().interpret(model)(_cont=lambda v: v)() == 123

    # cases where identifying the source of the lambda is impossible
    # should be caught
    with pytest.raises(ParsingError):
        model1, model2 = lambda : Bernoulli(.1).sample(), lambda : Bernoulli(.9).sample()
        CPSInterpreter().interpret(model1)
    
    # inspect.getsource is sensitive to what line a lambda is defined on
    # so this works
    model1, model2 = (
        lambda : Bernoulli(.1).sample(),
        lambda : Bernoulli(.9).sample()
    )
    CPSInterpreter().interpret(model1)

    # bytecode for inner and outer lambda are actually the same length
    # here, so we need to compare arguments
    model_maker = lambda p : lambda : Bernoulli(p).sample()
    model = model_maker(.5)
    CPSInterpreter().interpret(model)

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
    # This case is a bad one -- incorrect semantics.
    with pytest.raises(AssertionError) as err:
        def fn():
            def nested():
                return x
            x = 'bad'
            Bernoulli(0.5).sample() # Arbitrary. To put rest of function in continuation.
            x = 'good'
            return nested()
        check_trace(fn, [(Bernoulli(0.5), 0)], return_value='good')
    assert "assert 'bad' == 'good'" in str(err)

    # This case isn't quite as bad -- simply fails to run.
    with pytest.raises(NameError) as err:
        def fn():
            def nested():
                return x
            Bernoulli(0.5).sample() # Arbitrary. To put rest of function in continuation.
            x = 'good'
            return nested()
        check_trace(fn, [(Bernoulli(0.5), 0)], return_value='good')
    assert "name 'x' is not defined" in str(err)
