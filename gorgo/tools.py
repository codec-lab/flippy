import math

ISCLOSE_RTOL = 1e-5
ISCLOSE_ATOL = 1e-8

def isclose(a, b, *, rtol=ISCLOSE_RTOL, atol=ISCLOSE_ATOL):
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
