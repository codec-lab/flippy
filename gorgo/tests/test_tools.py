from gorgo.tools import isclose

def test_isclose():
    assert not isclose(1, 1+1e-1)
    assert not isclose(1, 1+1e-2)
    assert not isclose(1, 1+1e-3)
    assert not isclose(1, 1+1e-4)
    assert isclose(1, 1+1e-5)
    assert isclose(1, 1+1e-6)
    assert isclose(1, 1+1e-7)
    assert isclose(1, 1+1e-8)
