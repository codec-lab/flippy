import functools
import math
from typing import Callable, Sequence, TYPE_CHECKING, Union, TypeVar, overload
from gorgo.transforms import CPSTransform
from gorgo.inference import _distribution_from_inference, \
    SimpleEnumeration, Enumeration, SamplePrior, MetropolisHastings, LikelihoodWeighting
from gorgo.distributions import Categorical, Bernoulli, Distribution, Uniform, Element
from gorgo.distributions.random import default_rng
from gorgo.core import global_store
from gorgo.types import CPSCallable, Continuation
from gorgo.hashable import hashabledict
from gorgo.map import recursive_map

if TYPE_CHECKING:
    from gorgo.interpreter import CPSInterpreter, Stack

__all__ = [
    # Core API
    'infer',
    'keep_deterministic',
    'factor',
    'condition',
    'flip',
    'draw_from',
    'mem',
    'uniform',
    'map_observe',
    'default_rng',
    # Distributions
    'Categorical',
    'Bernoulli',
]

R = TypeVar('R')

# def keep_deterministic(fn: Callable) -> CPSCallable: #note that this is the real type signature
def keep_deterministic(fn: Callable[..., R]) -> Callable[..., R]:
    def continuation(*args, _cont: Continuation=None, _cps: 'CPSInterpreter'=None, _stack: 'Stack'=None, **kws):
        rv = fn(*args, **kws)
        if _cont is None:
            return rv
        else:
            return lambda : _cont(rv)
    setattr(continuation, CPSTransform.is_transformed_property, True)
    return continuation

def infer(
    func: Callable[..., Element]=None,
    method=SimpleEnumeration,
    cache_size=0,
    **kwargs
) -> Callable[..., Distribution[Element]]:
    if func is None:
        return functools.partial(infer, method=method, cache_size=cache_size, **kwargs)

    # After the wrapped function is CPS transformed, it will be evaluated.
    # If it is decorated with this function, the CPS-transformed function
    # will be passed in again. We simply return it.
    if CPSTransform.is_transformed(func):
        return func

    if isinstance(method, str):
        method = {
            'Enumeration': Enumeration,
            'SimpleEnumeration': SimpleEnumeration,
            'SamplePrior': SamplePrior,
            'MetropolisHastings': MetropolisHastings,
            'LikelihoodWeighting' : LikelihoodWeighting
        }[method]

    func = method(func, **kwargs)

    def wrapped(*args, _cont=None, _cps=None, **kws) -> Distribution[Element]:
        dist = func.run(*args, **kws)
        return _distribution_from_inference(dist)

    if cache_size > 0:
        wrapped = functools.lru_cache(maxsize=cache_size)(wrapped)
    wrapped = keep_deterministic(wrapped)
    return wrapped

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
        return 0

    def log_probability(self, element : float) -> float:
        #workaround for arbitrary scores
        return element

_factor_dist = FactorDistribution()

def factor(score):
    _factor_dist.observe(score)

def condition(cond: float):
    if cond == 0:
        _factor_dist.observe(-float("inf"))
    else:
        _factor_dist.observe(math.log(cond))

def flip(p=.5, name=None):
    return bool(Bernoulli(p).sample(name=name))

@keep_deterministic
def _draw_from_dist(n: Union[Sequence[Element], int]) -> Distribution[Element]:
    if isinstance(n, int):
        return Categorical(range(n))
    if hasattr(n, '__getitem__'):
        return Categorical(n)
    else:
        return Categorical(list(n))

@overload
def draw_from(n: int) -> int:
    ...
@overload
def draw_from(n: Sequence[Element]) -> Element:
    ...
def draw_from(n: Union[Sequence[Element], int]) -> Element:
    return _draw_from_dist(n).sample()

def mem(fn):
    def mem_wrapper(*args, **kws):
        key = (fn, args, tuple(sorted(kws.items())))
        kws = hashabledict(kws)
        if key in global_store:
            return global_store.get(key)
        else:
            value = fn(*args, **kws)
            global_store.set(key, value)
            return value
    return mem_wrapper

_uniform = Uniform()
def uniform():
    return _uniform.sample()

@keep_deterministic
def map_log_probability(distribution: Distribution[Element], values: Sequence[Element]) -> float:
    return sum(distribution.log_probability(i) for i in values)

def map_observe(distribution: Distribution[Element], values: Sequence[Element]) -> float:
    """
    Calculates the total log probability of a sequence of
    independent values from a distribution.
    """
    log_prob = map_log_probability(distribution, values)
    factor(log_prob)
    return log_prob
