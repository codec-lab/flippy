from typing import Sequence, Generic, TypeVar, Any, Callable, Hashable, Tuple
from itertools import combinations_with_replacement
from collections import Counter
import math
import random
import abc
from gorgo.tools import isclose
from functools import cached_property

Element = TypeVar("Element")

class Distribution(Generic[Element]):
    def prob(self, element : Element):
        return math.exp(self.log_probability(element))

    @abc.abstractmethod
    def sample(self, rng=random, name=None) -> Element:
        pass

    @abc.abstractmethod
    def log_probability(self, element : Element) -> float:
        pass

    def expected_value(self, func: Callable[[Element], Any] = lambda v : v) -> Any:
        raise NotImplementedError


class FiniteDistribution(Distribution):
    support: Sequence[Element]

    @cached_property
    def probabilities(self):
        return tuple(self.prob(e) for e in self.support)

    def isclose(self, other: "FiniteDistribution") -> bool:
        full_support = set(self.support) | set(other.support)
        return all(
            isclose(self.log_probability(s), other.log_probability(s))
            for s in full_support
        )

    def items(self):
        yield from zip(self.support, self.probabilities)

    def expected_value(self, func: Callable[[Element], Any] = lambda v : v) -> Any:
        return sum(
            p*func(s)
            for s, p in self.items()
        )

class Bernoulli(FiniteDistribution):
    support = (True, False)
    def __init__(self, p=.5):
        self.p = p
    def sample(self, rng=random, name=None) -> bool:
        if rng.random() <= self.p:
            return True
        return False
    def log_probability(self, element):
        return {
            True: math.log(self.p) if self.p != 0.0 else float('-inf'),
            False: math.log(1 - self.p) if self.p != 1.0 else float('-inf')
        }[element]


class Categorical(FiniteDistribution):
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

    @property
    def probabilities(self):
        return self._probabilities

    def sample(self, rng=random, name=None) -> Element:
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


class Multinomial(FiniteDistribution):
    def __init__(self, categorical_support, trials, *, probabilities=None, weights=None):
        self.categorical = Categorical(
            categorical_support,
            probabilities=probabilities,
            weights=weights
        )
        self.trials = trials

    def sample(self, rng=random, name=None) -> Tuple[int, ...]:
        samples = rng.choices(
            self.categorical.support,
            weights=self.categorical._probabilities,
            k=self.trials
        )
        counts = Counter(samples)
        return tuple(counts.get(i, 0) for i in self.categorical.support)

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

class Gaussian(Distribution):
    support = ClosedInterval(float('-inf'), float('inf'))
    def __init__(self, mean=0, sd=1):
        self.mean = mean
        self.sd = sd

    def sample(self, rng=random, name=None) -> float:
        return rng.gauss(self.mean, self.sd)

    def log_probability(self, element):
        prob = (
            math.e**(-.5*((element - self.mean)/self.sd)**2)
        )/(
            self.sd*(2*math.pi)**.5
        )
        return math.log(prob) if prob > 0. else float('-inf')

class Uniform(Distribution):
    def __init__(self, start=0, end=1):
        self.start = start
        self.end = end
        self.support = ClosedInterval(start, end)

    def sample(self, rng=random, name=None) -> float:
        return rng.uniform(self.start, self.end)

    def log_probability(self, element):
        if element in self.support:
            return math.log(1/(self.end - self.start))
        return float('-inf')

class Beta(Distribution):
    support = ClosedInterval(0, 1)
    def __init__(self, alpha=1, beta=1):
        self.a = alpha
        self.b = beta

    def sample(self, rng=random, name=None) -> float:
        return rng.betavariate(self.a, self.b)

    def log_probability(self, element):
        if element in self.support:
            num = (element**(self.a - 1))*(1 - element)**(self.b - 1)
            prob = num/beta_function(self.a, self.b)
            return math.log(prob) if prob != 0 else float('-inf')
        return float('-inf')

class Binomial(Distribution):
    def __init__(self, trials : int, p : float):
        self.trials = trials
        self.p = p
        assert 0 <= p <= 1
        self.support = tuple(range(0, self.trials + 1))

    def sample(self, rng=random, name=None) -> int:
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

class Geometric(Distribution):
    support = IntegerInterval(0, float('inf'))
    def __init__(self, p : float):
        self.p = p
        assert 0 <= p <= 1

    def sample(self, rng=random, name=None) -> int:
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

class Poisson(Distribution):
    support = IntegerInterval(0, float('inf'))
    def __init__(self, rate : float):
        self.rate = rate
        assert rate >= 0

    def sample(self, rng=random, name=None) -> int:
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

class BetaBinomial(FiniteDistribution):
    def __init__(self, trials : int, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.trials = trials
        self.support = tuple(range(0, self.trials + 1))

    def sample(self, rng=random, name=None) -> int:
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

    @cached_property
    def _enumerated_partitions(self):
        all_partitions = []
        for bins in combinations_with_replacement(range(self.total + 1), self.partitions - 1):
            partition = []
            for left, right in zip((0, ) + bins, bins + (self.total,)):
                partition.append(right - left)
            all_partitions.append(tuple(partition))
        return tuple(all_partitions)

    def __iter__(self):
        yield from self._enumerated_partitions

class Dirichlet(Distribution):
    def __init__(self, alphas):
        self.alphas = alphas

    @cached_property
    def support(self):
        return Simplex(len(self.alphas))

    def sample(self, rng=random, name=None) -> Tuple[float, ...]:
        e = [rng.gammavariate(a, 1) for a in self.alphas]
        tot = sum(e)
        return tuple(ei/tot for ei in e)

    def log_probability(self, vec):
        if vec in self.support:
            num = math.prod(v**(a - 1) for v, a in zip(vec, self.alphas))
            return math.log(num/beta_function(*self.alphas))
        return float('-inf')

class DirichletMultinomial(FiniteDistribution):
    def __init__(self, trials, alphas):
        self.alphas = alphas
        self.trials = trials

    @cached_property
    def support(self) -> OrderedIntegerPartitions:
        return OrderedIntegerPartitions(
            total=self.trials, partitions=len(self.alphas)
        )

    @cached_property
    def dirichlet(self) -> Dirichlet:
        return Dirichlet(self.alphas)

    def sample(self, rng=random, name=None) -> Tuple[int, ...]:
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
