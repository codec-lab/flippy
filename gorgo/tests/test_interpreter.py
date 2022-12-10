from gorgo.interpreter import *
from gorgo.core import Bernoulli, Multinomial
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
        return Multinomial(range(5)).sample()
    def OP():
        return Multinomial(['+', '*']).sample()
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
        m = ps.message
        assert isinstance(m, SampleMessage), (f'{trace_idx=}', m)
        assert isinstance(m.distribution, dist.__class__), (f'{trace_idx=}', m)
        assert m.distribution.__dict__ == dist.__dict__, (f'{trace_idx=}', m)
        ps = ps.step(value)

    m = ps.message
    assert isinstance(m, ReturnMessage)
    assert m.value == return_value

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
        (Multinomial(range(5)), 3),
    ], return_value=3)

    check_trace(algebra, [
        (Bernoulli(0.5), 0),
        (Multinomial(range(5)), 3),
        (Multinomial(['+', '*']), '*'),
        # subtree on right
        (Bernoulli(0.5), 0),
        (Multinomial(range(5)), 2),
        (Multinomial(['+', '*']), '+'),
        (Bernoulli(0.5), 1),
        (Multinomial(range(5)), 4),
    ], return_value=(3, '*', (2, '+', 4)))

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
    exception_line = 'raise __body_0__body_0__exc'

    # First, we just check that the traceback will render the code.
    formatted = ''.join(traceback.format_exception(exc, None, tb))
    assert exception_line in formatted

    # This is a more detailed check of the contents, ensuring the filename is correct in traceback.
    last_entry = traceback.extract_tb(tb)[-1]
    assert 'test_fn' in last_entry.filename
    assert hex(id(test_fn)).removeprefix('0x') in last_entry.filename
    assert last_entry.line == exception_line
