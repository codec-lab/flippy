import math
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Sequence, Set, Union, TYPE_CHECKING, Generic, Callable, TypeVar
from itertools import combinations_with_replacement, product
from functools import cached_property

from flippy.tools import isclose, ISCLOSE_RTOL, ISCLOSE_ATOL

if TYPE_CHECKING:
    from flippy.distributions.base import Distribution

Support = Union[
    Sequence,
    Set,
    "Set_"
]

Element = TypeVar('Element')

class Set_(Generic[Element]):
    def __contains__(self, ele: Element) -> bool:
        raise NotImplementedError

    def __pow__(self, exp: int):
        if not isinstance(exp, int) or exp < 0:
            raise ValueError("Exponent must be a non-negative integer")
        return ProductSet(*(self for _ in range(exp)))

@dataclass(frozen=True)
class Interval(Set_):
    start: float
    end: float
    left_open: bool = False
    right_open: bool = False

    def __contains__(self, ele: Element) -> bool:
        if isinstance(ele, (float, int)):
            if self.left_open:
                after_start = ele > self.start
            else:
                after_start = ele >= self.start
            if self.right_open:
                before_end = ele < self.end
            else:
                before_end = ele <= self.end
            return after_start and before_end
        return False

class Range(Set_):
    def __init__(self, *args):
        if len(args) == 1:
            self.start, self.end, self.step = 0, args[0], 1
        elif len(args) == 2:
            self.start, self.end, self.step = args[0], args[1], 1
        elif len(args) == 3:
            self.start, self.end, self.step = args
        else:
            raise ValueError("Range requires 1, 2, or 3 arguments: Range(stop), Range(start, stop), or Range(start, stop, step)")
        assert isinstance(self.start, int), "Start must be an integer"
        assert isinstance(self.end, int) or self.end == float('inf'), "End must be an integer"
        assert isinstance(self.step, int), "Step must be an integer"
        assert self.step > 0, "Step must be positive"

    def __contains__(self, ele: Element) -> bool:
        if not isinstance(ele, int):
            return False
        return (
            self.start <= ele < self.end and
            (ele - self.start) % self.step == 0
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.start}, {self.end}, {self.step})"

class ProductSet(Set_):
    def __init__(self, *seqs):
        self.seqs = seqs
    def __contains__(self, ele: Sequence[Element]) -> bool:
        return (
            isinstance(ele, Iterable) and
            len(ele) == len(self.seqs) and
            all(e in s for e, s in zip(ele, self.seqs))
        )
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(map(repr, self.seqs))})"
    def __iter__(self):
        yield from product(*self.seqs)
    def __len__(self):
        return math.prod([len(s) for s in self.seqs])

class UnionSet(Set_):
    def __init__(self, *sets: Set_):
        self.sets = sets
    def __contains__(self, ele: Element) -> bool:
        return any(ele in s for s in self.sets)
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(map(repr, self.sets))})"

class Simplex(Set_):
    def __init__(self, dimensions):
        self.dimensions = dimensions
    def __contains__(self, ele: Sequence[float]) -> bool:
        return (
            isinstance(ele, (tuple, list)) and \
            len(ele) == self.dimensions and \
            isclose(1.0, sum(ele))
        )
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dimensions})"


class OrderedIntegerPartitions(Set_):
    # https://en.wikipedia.org/wiki/Composition_(combinatorics)
    def __init__(self, total, partitions):
        self.total = total
        self.partitions = partitions

    def __contains__(self, ele: Sequence[int]) -> bool:
        return (
            isinstance(ele, (tuple, list)) and \
            all(isinstance(x, int) and x >= 0 for x in ele) and \
            len(ele) == self.partitions and \
            sum(ele) == self.total
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
