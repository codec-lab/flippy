from flippy.tools import isclose, LRUCache

def test_isclose():
    assert not isclose(1, 1+1e-1)
    assert not isclose(1, 1+1e-2)
    assert not isclose(1, 1+1e-3)
    assert not isclose(1, 1+1e-4)
    assert isclose(1, 1+1e-5)
    assert isclose(1, 1+1e-6)
    assert isclose(1, 1+1e-7)
    assert isclose(1, 1+1e-8)

def test_lru_cache():
    lru_inf = LRUCache(max_size=None)
    assert lru_inf.max_size == float('inf')

    lru_finite = LRUCache(max_size=2)
    lru_finite['a'] = 1
    lru_finite['b'] = 2
    lru_finite['c'] = 3
    assert set(lru_finite.keys()) == {'b', 'c'}
    lru_finite['d'] = 4
    assert set(lru_finite.keys()) == {'c', 'd'}
    lru_finite['c']
    lru_finite['e'] = 5
    assert set(lru_finite.keys()) == {'c', 'e'}
    lru_finite['e'] = 6
    assert lru_finite['e'] == 6
