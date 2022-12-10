from gorgo.core import Multinomial, Bernoulli

def test_distribution_isclose():
    # Same distribution.
    a = Multinomial(range(3))
    b = Multinomial(range(3))
    assert a.isclose(b)
    assert b.isclose(a)

    # Mismatched probabilities.
    a = Multinomial(range(3))
    b = Multinomial(range(3), probabilities=[.1, .4, .5])
    assert not a.isclose(b)
    assert not b.isclose(a)

    # Mismatched support, but probabilities match on non-zero support.
    a = Multinomial(range(2))
    b = Multinomial(range(3), probabilities=[.5, .5, 0])
    assert a.isclose(b)
    assert b.isclose(a)

    # Mismatched support.
    a = Multinomial(range(2))
    b = Multinomial(range(3))
    assert not a.isclose(b)
    assert not b.isclose(a)

    # Different distributions, but match.
    a = Multinomial([True, False])
    b = Bernoulli(0.5)
    assert a.isclose(b)
    assert b.isclose(a)
