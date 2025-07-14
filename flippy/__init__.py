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

- [Introductory tutorial](../tutorials/00-intro)

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
from flippy.distributions import Categorical, Bernoulli, Distribution, Uniform, Element
from flippy.distributions.random import default_rng
from flippy.distributions.base import _factor_dist
from flippy.core import global_store
from flippy.hashable import hashabledict
from flippy.map import recursive_map
from flippy.tools import LRUCache

from flippy.interpreter import CPSInterpreter

__all__ = [
    'infer',

    'flip',
    'draw_from',
    'uniform',

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

# Note if we use python 3.10+ we can use typing.ParamSpec
# so that combinators preserve the type signature of functions
R = TypeVar('R')

class DescriptorMixIn:
    """
    @private
    A mixin class that provides a descriptor interface for a callable object.
    """
    def __init__(self, wrapped_func):
        if inspect.ismethod(wrapped_func):
            raise ValueError("Cannot wrap a method outside the class namespace")
        self.is_classmethod = isinstance(wrapped_func, classmethod)
        self.is_staticmethod = isinstance(wrapped_func, staticmethod)

    def __call__(self, *args, _cont=None, _cps=None, _stack=None, **kws):
        raise NotImplementedError

    def __get__(self, obj, objtype=None):
        # if we wrapped a class method, we return a partial function with cls
        if self.is_classmethod:
            objtype = objtype if objtype is not None else type(obj)
            partial_call = functools.partial(self.__call__, objtype)
            setattr(partial_call, CPSTransform.is_transformed_property, True)
            return partial_call

        # if we wrapped a static method, we return the function itself
        if self.is_staticmethod:
            return self

        # if we wrapped a normal method, we return the function itself or
        # a partial function with obj
        if obj is None:
            return self
        partial_call = functools.partial(self.__call__, obj)
        setattr(partial_call, CPSTransform.is_transformed_property, True)
        return partial_call

class KeepDeterministicCallable(DescriptorMixIn):
    '''
    @private
    '''
    def __init__(self, func):
        DescriptorMixIn.__init__(self, func)
        self.wrapped_func = func
        if isinstance(func, (classmethod, staticmethod)):
            self.wrapped_func = func.__func__
        functools.update_wrapper(self, func)
        # This ensures that no subsequent compilation happens.
        setattr(self, CPSTransform.is_transformed_property, True)

    def __call__(self, *args, _cont=None, _cps=None, _stack=None, **kws):
        rv = self.wrapped_func(*args, **kws)
        if _cont is None:
            return rv
        else:
            return lambda : _cont(rv)


def keep_deterministic(fn: Callable[..., R]) -> Callable[..., R]:
    '''
    Decorator to interpret a function as deterministic Python.
    Any random sampling in the function will not be targeted for inference.

    This is helpful if the transform slows a function down, if a
    deterministic library is being called, or if a distribution is being
    directly computed.
    '''
    return KeepDeterministicCallable(fn)

def cps_transform_safe_decorator(dec: Callable) -> Callable:
    """
    A higher-order function that wraps a decorator so that it works with functions
    that are CPS transformed or will be CPS transformed.

    Functions are typically evaluated twice in the course of being CPS transformed.
    First, when they are initially declared in Python.
    Second, when being evaluated after being transformed.
    So, a function's decorators would typically be executed twice. However, this decorator
    can be used to mark a decorator so that it will not be executed after being CPS transformed.
    The resulting function will then be the transformed version of decorated function, without
    any further decoration after transformation.
    """
    @keep_deterministic
    def wrapped_decorator(fn=None, *args, **kws):
        if fn is None:
            return functools.partial(wrapped_decorator, *args, **kws)

        # This code mainly supports two cases:
        # 1. We need to CPS transform and wrap a function (e.g., when decorating in the module scope)
        # 2. We only need to wrap a function (e.g., when decorating in a nested function scope)

        # Case 1 is confusing because the transformation and wrapping occur in two different
        # calls to the current code. Specifically, we visit the current
        # code first from the module scope and then transform/wrap it using a CPSInterpreter
        # object. The CPSInterpreter object transforms the source code and then executes the current
        # code a second time on the transformed function (in an "exec" statement). This second
        # call to the current code does not need to transform the function, and this is indicated
        # by a _compile_mode flag associated with the CPSInterpreter class.

        # Case 2 is simpler because we only need to wrap the function. This occurs when the function
        # is nested inside another function that has already been CPS transformed.
        if CPSInterpreter._compile_mode:
            return fn
        wrapped_fn = dec(fn, *args, **kws)
        return wrapped_fn
    wrapped_decorator = functools.wraps(dec)(wrapped_decorator)
    return wrapped_decorator

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
        if isinstance(func, (classmethod, staticmethod)):
            func = func.__func__
        if not CPSTransform.is_transformed(func):
            func = CPSInterpreter().non_cps_callable_to_cps_callable(func)
        self.inference_alg = self.method(func, **self.kwargs)
        setattr(self, CPSTransform.is_transformed_property, True)

    def __call__(self, *args, _cont=None, _cps=None, _stack=None, **kws) -> Distribution[Element]:
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
