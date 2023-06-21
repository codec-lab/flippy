from typing import Sequence, Set, Union
from itertools import combinations_with_replacement
from gorgo.tools import isclose, ISCLOSE_RTOL, ISCLOSE_ATOL
from functools import cached_property

Support = Union[Sequence, Set, 'ClosedInterval', 'IntegerInterval', 'Simplex', 'OrderedIntegerPartitions']

class ClosedInterval:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __contains__(self, ele):
        return self.start <= ele <= self.end
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.start}, {self.end})"


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
