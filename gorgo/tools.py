import math

ISCLOSE_RTOL = 1e-5
ISCLOSE_ATOL = 1e-8

def isclose(a, b, *, rtol=ISCLOSE_RTOL, atol=ISCLOSE_ATOL):
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)

def logsumexp(*args):
    max_arg = max(args)
    return max_arg + math.log(sum(math.exp(arg - max_arg) for arg in args))

def softmax_dict(d: dict):
    max_value = max(d.values())
    keys, values = zip(*d.items())
    exp_values = [math.exp(v - max_value) for v in values]
    total = sum(exp_values)
    return {k: v/total for k, v in zip(keys, exp_values)}
