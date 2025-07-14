from flippy import infer, flip, keep_deterministic, Bernoulli, mem, factor
from flippy.map import independent_map
from flippy.inference.simpleenumeration import SimpleEnumeration
from flippy.inference.enumeration import Enumeration

def test_independent_map():
    iterator = (.1, .2, .3, .4, .5, .6, .7, .8, .9)
    def f1():
        def g(i):
            return flip(i) + 1
        x = independent_map(g, iterator)
        return x

    def f2():
        def g(i):
            return flip(i) + 1
        x = tuple([g(i) for i in iterator])
        return x

    res1 = SimpleEnumeration(f1).run()
    res2 = SimpleEnumeration(f2).run()
    res3 = Enumeration(f1).run()
    assert res1.isclose(res2)
    assert res1.isclose(res3)

def test_independent_map_with_mem():
    def f(n):
        @mem
        def h(i):
            return flip(0.5)
        def g(p):
            i = flip(p) + flip(p)
            return h(i)
        return independent_map(g, tuple([i/(n - 1) for i in range(n)]))

    e_res = SimpleEnumeration(f).run(n=3)
    ge_res = Enumeration(f).run(n=3)
    assert e_res.isclose(ge_res)
