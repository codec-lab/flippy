from typing import Sequence, Generic, TypeVar, Any, Callable, Hashable, Tuple
import math
import random
import abc
from gorgo.tools import isclose
from gorgo.funcutils import cached_property

############################################
#  Sampling and observations
############################################

Element = TypeVar("Element")

class Distribution(Generic[Element]):
    support: Sequence[Element]
    def log_probability(self, element : Element) -> float:
        pass
    def sample(self, name=None) -> Element:
        pass
    def isclose(self, other: "Distribution") -> bool:
        full_support = set(self.support) | set(other.support)
        return all(
            isclose(self.log_probability(s), other.log_probability(s))
            for s in full_support
        )

class StochasticPrimitive(Distribution):
    support : Sequence
    @abc.abstractmethod
    def __call__(self, *params, rng=random):
        pass
    def sample(self, rng=random, name=None):
        return self(rng=rng)

class Bernoulli(StochasticPrimitive):
    support = (True, False)
    def __init__(self, p=.5):
        self.p = p
    def __call__(self, rng=random):
        if rng.random() <= self.p:
            return True
        return False
    def log_probability(self, element):
        return {
            True: math.log(self.p),
            False: math.log(1 - self.p) if self.p != 1.0 else float('-inf')
        }[element]

class Multinomial(StochasticPrimitive):
    def __init__(self, support, *, probabilities=None, weights=None):
        if probabilities is not None:
            assert isclose(sum(probabilities), 1)
        elif weights is not None:
            probabilities = [w / sum(weights) for w in weights]
            del weights
        else:
            probabilities = [1/len(support) for _ in support]
        self.support = support
        self._probabilities = probabilities

    def __call__(self, rng=random):
        return rng.choices(self.support, weights=self._probabilities, k=1)[0]

    def log_probability(self, element):
        try:
            return math.log(self._probabilities[self.support.index(element)])
        except ValueError:
            return float('-inf')

    def __repr__(self):
        return repr({
            s: self.log_probability(s)
            for s in self.support
        })

class ClosedInterval:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __contains__(self, ele):
        return self.start <= ele <= self.end

class IntegerInterval(ClosedInterval):
    def __contains__(self, ele):
        if ele == int(ele):
            return self.start <= ele <= self.end
        return False

def beta_function(a, b):
    num = math.gamma(a)*math.gamma(b)
    den = math.gamma(a + b)
    return num/den
    
class Gaussian(StochasticPrimitive):
    support = ClosedInterval(float('-inf'), float('inf'))
    def __init__(self, mean=0, sd=1):
        self.mean = mean
        self.sd = sd 
        
    def __call__(self, rng=random):
        return rng.gauss(self.mean, self.sd)
    
    def log_probability(self, element):
        prob = (
            math.e**(-.5*((element - self.mean)/self.sd)**2)
        )/(
            self.sd*(2*math.pi)**.5
        )
        return math.log(prob) if prob > 0. else float('-inf')

class Uniform(StochasticPrimitive):
    def __init__(self, start=0, end=1):
        self.start = start
        self.end = end
        self.support = ClosedInterval(start, end)
        
    def __call__(self, rng=random):
        return rng.uniform(self.start, self.end)
    
    def log_probability(self, element):
        if element in self.support:
            return math.log(1/(self.end - self.start))
        return float('-inf')
    
class Beta(StochasticPrimitive):
    support = ClosedInterval(0, 1)
    def __init__(self, alpha=1, beta=1):
        self.a = alpha
        self.b = beta
        
    def __call__(self, rng=random):
        return rng.betavariate(self.a, self.b)
    
    def log_probability(self, element):
        if element in self.support:
            num = (element**(self.a - 1))*(1 - element)**(self.b - 1)
            return math.log(num/beta_function(self.a, self.b))
        return float('-inf')

class Binomial(StochasticPrimitive):
    def __init__(self, n : int, p : float):
        self.n = n
        self.p = p
        assert 0 <= p <= 1
        self.support = tuple(range(0, self.n + 1))
        
    def __call__(self, rng=random):
        return sum(rng.random() < self.p for _ in range(self.n))
    
    def log_probability(self, element):
        if element in self.support:
            prob = (
                math.comb(self.n, element)
            )*(
                self.p**element
            )*(
                (1 - self.p)**(self.n - element)
            )
            return math.log(prob)
        return float('-inf')
    
class Geometric(StochasticPrimitive):
    support = IntegerInterval(0, float('inf'))
    def __init__(self, p : float):
        self.p = p
        assert 0 <= p <= 1
        
    def __call__(self, rng=random):
        i = 0
        while rng.random() >= self.p:
            i += 1
        return i
    
    def log_probability(self, element):
        if element in self.support:
            return math.log(
                ((1 - self.p)**(element))*self.p
            )
        return float('-inf')
    
class BetaBinomial(StochasticPrimitive):
    def __init__(self, n : int, alpha=1, beta=1):
        self.a = alpha
        self.b = beta
        self.n = n
        self.support = tuple(range(0, self.n + 1))
        
    def __call__(self, rng=random):
        p = rng.betavariate(self.a, self.b)
        return sum(rng.random() < p for _ in range(self.n))
    
    def log_probability(self, element):
        if element in self.support:
            prob = math.comb(self.n, element)*(
                (
                    beta_function(element+self.a, self.n - element + self.b)
                )/(
                    beta_function(self.a, self.b)
                )
            )
            return math.log(prob)
        return float('-inf')

# TODO: finalize this interface
class ObservationStatement:
    def __call__(
        self,
        distribution,
        value,
    ):
        pass
observe = ObservationStatement()

############################################
#  Program State
############################################

from collections import namedtuple
StackFrame = namedtuple("StackFrame", "func_src lineno locals")

class ProgramState:
    def __init__(
        self,
        continuation,
        name: Hashable = None,
        stack: Tuple[StackFrame] = None
    ):
        self.continuation = continuation
        self._name = name
        self.stack = stack

    def step(self, *args, **kws):
        thunk = self.continuation(*args, **kws)
        while True:
            next_ = thunk()
            if isinstance(next_, ProgramState):
                return next_
            thunk = next_

    @cached_property
    def name(self):
        if self._name is not None:
            return self._name
        if self.stack is None:
            return None
        return tuple((frame.func_src, frame.lineno) for frame in self.stack)

class InitialState(ProgramState):
    pass

class ObserveState(ProgramState):
    def __init__(
        self,
        continuation: Callable[[], Callable],
        distribution: Distribution,
        value: Any,
        name: Hashable,
        stack: Tuple[StackFrame] 
    ):
        super().__init__(
            continuation=continuation,
            name=name,
            stack=stack
        )
        self.distribution = distribution
        self.value = value

class SampleState(ProgramState):
    def __init__(
        self,
        continuation: Callable[[], Callable],
        distribution: Distribution,
        name: Hashable,
        stack: Tuple[StackFrame]
    ):
        super().__init__(
            continuation=continuation,
            name=name,
            stack=stack
        )
        self.distribution = distribution

class ReturnState(ProgramState):
    def __init__(self, value: Any):
        self.value = value

    def step(self, *args, **kws):
        raise ValueError