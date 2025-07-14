import math
from collections.abc import Iterable
from typing import Sequence, Set, Union, TYPE_CHECKING
from itertools import combinations_with_replacement, product
from flippy.tools import isclose, ISCLOSE_RTOL, ISCLOSE_ATOL
from functools import cached_property

if TYPE_CHECKING:
    from flippy.distributions.base import Distribution

Support = Union[
    Sequence,
    Set,
    'ClosedInterval',
    'IntegerInterval',
    'Simplex',
    'OrderedIntegerPartitions',
    'CrossProduct'
]

class ClosedInterval:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __contains__(self, ele):
        return isinstance(ele, (float, int)) and (self.start <= ele <= self.end)
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.start}, {self.end})"


class CrossProduct:
    def __init__(self, *seqs):
        self.seqs = seqs
    def __contains__(self, vec):
        return len(vec) == len(self.seqs) and all(e in s for e, s in zip(vec, self.seqs))
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(map(repr, self.seqs))})"
    def __iter__(self):
        yield from product(*self.seqs)
    def __len__(self):
        return math.prod([len(s) for s in self.seqs])


class IntegerInterval(ClosedInterval):
    def __contains__(self, ele):
        if ele == int(ele):
            return self.start <= ele <= self.end
        return False
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.start}, {self.end})"

class Simplex:
    def __init__(self, dimensions):
        self.dimensions = dimensions
    def __contains__(self, vec):
        return (
            isinstance(vec, Iterable) and \
            len(vec) == self.dimensions and \
            isclose(1.0, sum(vec)) and \
            all((not isclose(0.0, e)) and (0 < e <= 1) for e in vec)
        )
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dimensions})"


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(total={self.total}, partitions={self.partitions})"

class MixtureSupport:
    def __init__(self, distributions : Sequence['Distribution']):
        self.distributions = distributions

    def __contains__(self, element):
        return any(element in d.support for d in self.distributions)
