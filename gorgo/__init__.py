import functools
from typing import Callable
from gorgo.transforms import CPSTransform
from gorgo.inference import _distribution_from_inference, \
    Enumeration, SamplePrior, MetropolisHastings, LikelihoodWeighting
from gorgo.distributions import Categorical, Bernoulli, Distribution
from gorgo.core import global_store

__all__ = [
    # Core API
    'infer',
    'keep_deterministic',
    'factor',
    'condition',
    'flip',
    'draw',
    'mem',
    # Distributions
    'Categorical',
    'Bernoulli',
]

def keep_deterministic(fn):
    def wrapped(*args, _cont=None, _cps=None, _stack=None, **kws):
        rv = fn(*args, **kws)
        if _cont is None:
            return rv
        else:
            return lambda : _cont(rv)
    setattr(wrapped, CPSTransform.is_transformed_property, True)
    return wrapped

def infer(func=None, method=Enumeration, **kwargs) -> Callable[..., Distribution]:
    if func is None:
        return functools.partial(infer, method=method, **kwargs)

    # After the wrapped function is CPS transformed, it will be evaluated.
    # If it is decorated with this function, the CPS-transformed function
    # will be passed in again. We simply return it.
    if CPSTransform.is_transformed(func):
        return func

    if isinstance(method, str):
        method = {
            'Enumeration': Enumeration,
            'SamplePrior': SamplePrior,
            'MetropolisHastings': MetropolisHastings,
            'LikelihoodWeighting' : LikelihoodWeighting
        }[method]

    func = method(func, **kwargs)

    @keep_deterministic
    def wrapped(*args, _cont=None, _cps=None, **kws):
        dist = func.run(*args, **kws)
        return _distribution_from_inference(dist)

    return wrapped

def recursive_map(fn, iter):
    if not iter:
        return []
    return [fn(iter[0])] + recursive_map(fn, iter[1:])

def recursive_filter(fn, iter):
    if not iter:
        return []
    if fn(iter[0]):
        head = [iter[0]]
    else:
        head = []
    return head + recursive_filter(fn, iter[1:])

def recursive_reduce(fn, iter, initializer):
    if len(iter) == 0:
        return initializer
    return recursive_reduce(fn, iter[1:], fn(initializer, iter[0]))

class FactorDistribution(Distribution):
    def __init__(self):
        pass

    def sample(self, rng, name):
        pass

    def log_probability(self, element : float) -> float:
        #workaround for arbitrary scores
        return element

factor_dist = FactorDistribution()

def factor(score):
    factor_dist.observe(score)

def condition(cond):
    factor_dist.observe(0 if cond else -float("inf"))

def flip(p=.5):
    return Bernoulli(p).sample()

def draw_from(n):
    return Categorical(range(n)).sample()

def mem(fn):
    def wrapped(*args, **kws):
        key = (fn, args, tuple(kws.items()))
        if global_store.includes(key):
            return global_store.read(key)
        else:
            value = fn(*args, **kws)
            global_store.write(key, value)
            return value
    return wrapped
