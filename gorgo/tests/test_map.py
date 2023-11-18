from gorgo import infer, flip, keep_deterministic, Bernoulli, mem, factor
from gorgo.map import independent_map
from gorgo.inference import Enumeration

def test_independent_map():
    iterator = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    def f1():
        def g(i):
            return flip(i) + 1
        x = independent_map(g, iterator)
        return x

    def f2():
        def g(i):
            return flip(i) + 1
        x = [g(i) for i in iterator]
        return x

    res1 = Enumeration(f1).run()
    res2 = Enumeration(f2).run()
    assert res1.isclose(res2)
