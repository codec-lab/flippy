'''
FlipPy is a library for specifying probabilistic programs.

# Quick start

```bash
pip install flippy-lang
```

# Example: Sum of bernoullis

```python
@infer
def fn():
    x = flip(0.5)
    y = flip(0.5)
    return x + y

fn() # Distribution({0: 0.25, 1: 0.5, 2: 0.25})
```

# Documentation

Here is the documentation for writing models in FlipPy.
- The core API for declaring a model ([link](#api))
- Specifying distributions ([link](flippy/distributions))
- Selecting inference algorithms ([link](flippy/inference))

# Tutorials

- [Introductory tutorial](tutorials/00-intro)
- [Rational Speech Acts (RSA)](tutorials/01-RSA)
- [Language of Thought (LoT)](tutorials/02-LoT)
- [Hidden Markov Models (HMMs)](tutorials/03-HMMs)

# API

'''

import functools
import math
import inspect
from typing import Callable, Sequence, Union, TypeVar, overload, Generic
from flippy.transforms import CPSTransform
from flippy.inference import \
    SimpleEnumeration, Enumeration, SamplePrior, MetropolisHastings, \
    LikelihoodWeighting, InferenceAlgorithm
from flippy.distributions import Categorical, Bernoulli, Distribution, Uniform,\
    Element, Normal
from flippy.distributions.random import default_rng
from flippy.distributions.base import _factor_dist
from flippy.core import global_store
from flippy.hashable import hashabledict
from flippy.map import recursive_map
from flippy.tools import LRUCache

from flippy.interpreter import CPSInterpreter, keep_deterministic, \
    cps_transform_safe_decorator, DescriptorMixIn

__all__ = [
    'infer',

    'flip',
    'draw_from',
    'uniform',
    'normal',

    'factor',
    'condition',
    'map_observe',

    'mem',

    'keep_deterministic',

    # submodules

    # Model specification
    'distributions',
    # Inference algorithms
    'inference',

    # Execution model
    'core',
    'callentryexit',
    'map',
]

class InferCallable(Generic[Element], DescriptorMixIn):
    '''
    @private
    '''
    def __init__(
        self,
        func: Callable[..., Element],
        method : Union[type[InferenceAlgorithm], str] = "Enumeration",
        cache_size=0,
        **kwargs
    ):
        DescriptorMixIn.__init__(self, func)

        if isinstance(method, str):
            method : type[InferenceAlgorithm] = {
                'Enumeration': Enumeration,
                'SimpleEnumeration': SimpleEnumeration,
                'SamplePrior': SamplePrior,
                'MetropolisHastings': MetropolisHastings,
                'LikelihoodWeighting' : LikelihoodWeighting
            }[method]
        self.cache_size = cache_size
        self.cache = LRUCache(cache_size)
        self.method = method
        self.kwargs = kwargs
        self.func = func
        self.inference_alg = None
        setattr(self, CPSTransform.is_transformed_property, True)

    def _lazy_init(self):
        if self.inference_alg is not None:
            return
        func = self.func
        if isinstance(func, (classmethod, staticmethod)):
            func = func.__func__
        if not CPSTransform.is_transformed(func):
            func = CPSInterpreter().non_cps_callable_to_cps_callable(func)
        self.inference_alg = self.method(func, **self.kwargs)

        if not self.inference_alg.is_cachable:
            self.cache_size = 0

    def __call__(self, *args, _cont=None, _cps=None, _stack=None, **kws) -> Distribution[Element]:
        self._lazy_init()
        if self.cache_size > 0:
            kws_tuple = tuple(sorted(kws.items()))
            if (args, kws_tuple) in self.cache:
                dist = self.cache[args, kws_tuple]
            else:
                dist = self.inference_alg.run(*args, **kws)
                self.cache[args, kws_tuple] = dist
        else:
            dist = self.inference_alg.run(*args, **kws)
        if _cont is None:
            return dist
        else:
            return lambda : _cont(dist)

def infer(
    func: Callable[..., Element]=None,
    method=Enumeration,
    cache_size=1024,
    **kwargs
) -> InferCallable[Element]:
    '''
    Turns a function into a stochastic function, that represents a posterior distribution.

    This is the main interface for performing inference in FlipPy.

    - `method` specifies the inference method. Defaults to `Enumeration`.
    - `**kwargs` are keyword arguments passed to the inference method.
    '''
    return InferCallable(func, method, cache_size, **kwargs)
infer = cps_transform_safe_decorator(infer)

# type hints for infer - if we can use ParamSpecs this will be cleaner
InferenceType = Callable[[Callable[..., Element]], InferCallable[Element]]
infer : Callable[..., Union[InferCallable, InferenceType]]

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

def factor(score):
    '''
    Adds a real-valued `score` to the weight of the current trace.
    '''
    _factor_dist.observe(score)

def condition(cond: float):
    '''
    Used for conditioning statements. When `cond` is a boolean, this behaves like
    typical conditioning.

    - `cond` is a non-negative multiplicative weight for the conditioning. When zero,
        the trace is assigned zero probability.
    '''
    if cond == 0:
        _factor_dist.observe(-float("inf"))
    else:
        _factor_dist.observe(math.log(cond))

def flip(p=.5, name=None):
    '''
    Samples from a Bernoulli distribution with probability `p`.
    '''
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
    '''
    Samples uniformly from `n` when it is a sequence.
    When `n` is an integer, a sample is drawn from `range(n)`.
    '''
    return _draw_from_dist(n).sample()

def mem(fn: Callable[..., Element]) -> Callable[..., Element]:
    '''
    Turns a function into a stochastically memoized function.
    Stores information in trace-specific storage.
    '''
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
mem = cps_transform_safe_decorator(mem)

_uniform = Uniform()
def uniform():
    '''
    Samples from a uniform distribution over the interval $[0, 1]$.
    '''
    return _uniform.sample()

_normal = Normal(0, 1)
def normal(mean=0, std=1):
    '''
    Samples from a standard normal distribution.
    '''
    return mean + std * _normal.sample()

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
