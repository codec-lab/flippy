import math

def isclose(a, b, *, rtol=1e-5, atol=1e-8):
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
