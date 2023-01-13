from typing import Sequence, Generic, TypeVar, Any, Callable, Hashable, Tuple
from collections import Counter
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
    def prob(self, element : Element):
        return math.exp(self.log_probability(element))
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
            True: math.log(self.p) if self.p != 0.0 else float('-inf'),
            False: math.log(1 - self.p) if self.p != 1.0 else float('-inf')
        }[element]

class Categorical(StochasticPrimitive):
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
    
    def items(self):
        yield from zip(self.support, self._probabilities)

class Multinomial(StochasticPrimitive):
    def __init__(self, categorical_support, trials, *, probabilities=None, weights=None):
        self.categorical = Categorical(
            categorical_support, 
            probabilities=probabilities,
            weights=weights
        )
        self.trials = trials

    def __call__(self, rng=random):
        samples = rng.choices(
            self.categorical.support,
            weights=self.categorical._probabilities,
            k=self.trials
        )
        counts = Counter(samples)
        return tuple(counts.get(i, 0) for i in range(len(self.categorical.support)))

    @cached_property
    def support(self):
        return OrderedIntegerPartitions(
            total=self.trials, partitions=len(self.categorical.support)
        )
    
    def log_probability(self, vec):
        if vec in self.support:
            probs = self.categorical._probabilities
            num1 = math.gamma(self.trials + 1)
            num2 = math.prod(p**x for p, x in zip(probs, vec))
            den = math.prod(math.gamma(x + 1) for x in vec)
            return math.log((num1*num2)/den)
        return float('-inf')

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

def beta_function(*alphas):
    num = math.prod(math.gamma(a) for a in alphas)
    den = math.gamma(sum(alphas))
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
            prob = num/beta_function(self.a, self.b)
            return math.log(prob) if prob != 0 else float('-inf')
        return float('-inf')

class Binomial(StochasticPrimitive):
    def __init__(self, trials : int, p : float):
        self.trials = trials
        self.p = p
        assert 0 <= p <= 1
        self.support = tuple(range(0, self.trials + 1))
        
    def __call__(self, rng=random):
        return sum(rng.random() < self.p for _ in range(self.trials))
    
    def log_probability(self, element):
        if element in self.support:
            prob = (
                math.comb(self.trials, element)
            )*(
                self.p**element
            )*(
                (1 - self.p)**(self.trials - element)
            )
            return math.log(prob) if prob != 0 else float('-inf')
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

class Poisson(StochasticPrimitive):
    support = IntegerInterval(0, float('inf'))
    def __init__(self, rate : float):
        self.rate = rate
        assert rate >= 0
        
    def __call__(self, rng=random):
        p, k, L = 1, 0, math.exp(-self.rate)
        while p > L:
            k += 1
            p = rng.random()*p
        return k - 1
    
    def log_probability(self, k):
        if k not in self.support:
            return float('-inf')
        prob = (self.rate**k)*math.exp(-self.rate)/math.factorial(k)
        return math.log(prob)
    
class BetaBinomial(StochasticPrimitive):
    def __init__(self, trials : int, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.trials = trials
        self.support = tuple(range(0, self.trials + 1))
        
    def __call__(self, rng=random):
        p = rng.betavariate(self.alpha, self.beta)
        return sum(rng.random() < p for _ in range(self.trials))
    
    def log_probability(self, element):
        if element in self.support:
            prob = math.comb(self.trials, element)*(
                (
                    beta_function(element+self.alpha, self.trials - element + self.beta)
                )/(
                    beta_function(self.alpha, self.beta)
                )
            )
            return math.log(prob)
        return float('-inf')

class Simplex:
    def __init__(self, dimensions):
        self.dimensions = dimensions
    def __contains__(self, vec):
        return (
            len(vec) == self.dimensions and \
            isclose(1.0, sum(vec)) and \
            all((not isclose(0.0, e)) and (0 < e <= 1) for e in vec)
        )

class OrderedIntegerPartitions:
    # https://en.wikipedia.org/wiki/Composition_(combinatorics)
    def __init__(self, total, partitions):
        self.total = total
        self.partitions = partitions
    def __contains__(self, vec):
        return (
            len(vec) == self.partitions and \
            sum(vec) == self.total
        )
    
class Dirichlet(StochasticPrimitive):
    def __init__(self, alphas):
        self.alphas = alphas
    
    @cached_property
    def support(self):
        return Simplex(len(self.alphas))
        
    def __call__(self, rng=random):
        e = [rng.gammavariate(a, 1) for a in self.alphas]
        tot = sum(e)
        return tuple(ei/tot for ei in e)
    
    def log_probability(self, vec):
        if vec in self.support:
            num = math.prod(v**(a - 1) for v, a in zip(vec, self.alphas))
            return math.log(num/beta_function(*self.alphas))
        return float('-inf')

class DirichletMultinomial(StochasticPrimitive):
    def __init__(self, trials, alphas):
        self.alphas = alphas
        self.trials = trials
    
    @cached_property
    def support(self):
        return OrderedIntegerPartitions(
            total=self.trials, partitions=len(self.alphas)
        )

    @cached_property
    def dirichlet(self) -> Dirichlet:
        return Dirichlet(self.alphas)
        
    def __call__(self, rng=random):
        ps = self.dirichlet.sample(rng=rng)
        samples = rng.choices(range(len(self.alphas)), weights=ps, k=self.n)
        counts = Counter(samples)
        return tuple(counts.get(i, 0) for i in range(len(self.alphas)))
    
    def log_probability(self, vec):
        if vec not in self.support:
            return float('-inf')
        assert len(vec) == len(self.alphas)
        num = self.trials*beta_function(sum(self.alphas), self.trials)
        den = math.prod(
            x*beta_function(a, x) for a, x in zip(self.alphas, vec)
            if x > 0
        )
        return math.log(num/den)

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