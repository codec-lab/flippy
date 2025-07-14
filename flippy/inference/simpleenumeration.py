import heapq
from collections import defaultdict, Counter
import dataclasses
from typing import Any, Union, List, Tuple, Dict, Set, Tuple, Sequence
from itertools import product

from flippy.core import ProgramState, ReturnState, SampleState, ObserveState
from flippy.interpreter import CPSInterpreter
from flippy.distributions import Categorical
from flippy.tools import logsumexp, softmax_dict
from flippy.map import MapEnter
from flippy.callentryexit import EnterCallState, ExitCallState
from flippy.types import Element
from flippy.inference.inference import InferenceAlgorithm

@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=dataclasses.field(compare=False)

@dataclasses.dataclass(frozen=True)
class ProgramStateRecord:
    kind: type
    name: str

@dataclasses.dataclass
class EnumerationStats:
    states_visited: list[ProgramStateRecord] = dataclasses.field(default_factory=list)

    def site_counts(self):
        return Counter(self.states_visited)

class SimpleEnumeration(InferenceAlgorithm[Element]):
    def __init__(self, function, max_executions=float('inf')):
        self.function = function
        self.max_executions = max_executions
        self._stats = None

    def enumerate_tree(
        self,
        ps: ProgramState,
        max_executions: int,
    ) -> Dict[Any, float]:
        frontier: List[PrioritizedItem] = []
        return_scores = {}
        executions = 0
        heapq.heappush(frontier, PrioritizedItem(0, ps))
        while len(frontier) > 0:
            if executions >= max_executions:
                break
            item = heapq.heappop(frontier)
            cum_weight = -item.priority
            ps = item.item
            if isinstance(ps, SampleState):
                for value in ps.distribution.support:
                    weight = ps.distribution.log_probability(value)
                    if weight > float('-inf'):
                        new_ps = ps.step(value)
                        heapq.heappush(frontier, PrioritizedItem(-(cum_weight + weight), new_ps))
            elif isinstance(ps, ObserveState):
                value = ps.value
                weight = ps.distribution.log_probability(value)
                if weight > float('-inf'):
                    new_ps = ps.step()
                    heapq.heappush(frontier, PrioritizedItem(-(cum_weight + weight), new_ps))
            elif isinstance(ps, ReturnState):
                return_scores[ps.value] = logsumexp(
                    return_scores.get(ps.value, float('-inf')),
                    cum_weight
                )
                executions += 1
            elif isinstance(ps, (EnterCallState, ExitCallState, MapEnter)):
                new_ps = ps.step()
                heapq.heappush(frontier, PrioritizedItem(-cum_weight, new_ps))
            else:
                raise ValueError(f"Unrecognized program state message, {ps}")
            if self._stats is not None:
                self._stats.states_visited.append(ProgramStateRecord(ps.__class__, ps.name))
        return return_scores

    def run(self, *args, **kws) -> Categorical[Element]:
        init_ps = CPSInterpreter().initial_program_state(self.function)
        ps = init_ps.step(*args, **kws)
        return_scores = self.enumerate_tree(ps, self.max_executions)
        if len(return_scores) == 0:
            raise ValueError("No return states encountered during enumeration")
        return Categorical.from_dict(softmax_dict(return_scores))

    def _run_with_stats(self, *args, **kws) -> Tuple[Categorical, EnumerationStats]:
        self._stats = EnumerationStats()
        result = self.run(*args, **kws)
        self._stats, stats = None, self._stats
        return result, stats
