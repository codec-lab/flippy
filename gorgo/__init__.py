import functools
from gorgo.transforms import CPSTransform
from gorgo.inference import Enumeration, _distribution_from_inference
from gorgo.core import Multinomial, Bernoulli, observe

__all__ = [
    # Core API
    'infer',
    'keep_deterministic',
    'observe',
    # Distributions
    'Multinomial',
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

def infer(func=None, method=Enumeration, **kwargs):
    if func is None:
        return functools.partial(infer, **kwargs)

    # After the wrapped function is CPS transformed, it will be evaluated.
    # If it is decorated with this function, the CPS-transformed function
    # will be passed in again. We simply return it.
    if CPSTransform.is_transformed(func):
        return func

    if isinstance(method, str):
        method = {
            'Enumeration': Enumeration,
        }[method]

    func = method(func, **kwargs)

    @keep_deterministic
    def wrapped(*args, _cont=None, _cps=None, **kws):
        dist = func.run(*args, **kws)
        return _distribution_from_inference(dist)

    return wrapped
