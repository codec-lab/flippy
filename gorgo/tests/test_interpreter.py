from gorgo.interpreter import *
from gorgo.core import Bernoulli, Multinomial
import pytest

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
        return Multinomial(['+', '*'])
    def EQ():
        if flip():
            return NUM()
        else:
            return (NUM(), OP(), EQ())
    return EQ()

def check_trace(func, trace):
    return_value = trace.pop()

    ps = CPSInterpreter().initial_program_state(func)
    ps = ps.step(*trace.pop(0))

    for (dist, value) in trace:
        m = ps.message
        assert isinstance(m, SampleMessage)
        assert isinstance(m.distribution, dist.__class__)
        assert m.distribution.__dict__ == dist.__dict__
        ps = ps.step(value)

    m = ps.message
    assert isinstance(m, ReturnMessage)
    assert m.value == return_value

def test_interpreter():
    check_trace(geometric, [
        (0.9,),
        (Bernoulli(0.9), 0),
        0,
    ])

    check_trace(geometric, [
        (0.8,),
        (Bernoulli(0.8), 1),
        (Bernoulli(0.8), 1),
        (Bernoulli(0.8), 0),
        2,
    ])

    check_trace(algebra, [
        [],
        (Bernoulli(0.5), 1),
        (Multinomial(range(5)), 3),
        3,
    ])

    # TODO: fix this
    # check_trace(algebra, [
    #     [],
    #     (Bernoulli(0.5), 0),
    #     (Multinomial(range(5)), 3),
    #     (Bernoulli(0.5), 0),
    #     (Multinomial(range(5)), 2),
    #     (Bernoulli(0.5), 1),
    #     (Multinomial(['+', '*']), '*'),
    #     # (Bernoulli(0.5), 1),
    #     (3, '*', 2),
    # ])

def test_check_exception():
    def test_fn():
        raise Exception('expected exception')
        return 3
    with pytest.raises(Exception) as e:
        check_trace(test_fn, [(), None])
    assert 'expected exception' in str(e)
