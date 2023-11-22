import heapq
import dataclasses
from typing import Any, Union, List, Dict, Tuple, Sequence
from itertools import product

from gorgo.core import ProgramState, ReturnState, SampleState, ObserveState
from gorgo.interpreter import CPSInterpreter
from gorgo.distributions import Categorical
from gorgo.tools import logsumexp, softmax_dict
from gorgo.map import MapIterStart, MapIterEnd

@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=dataclasses.field(compare=False)

@dataclasses.dataclass
class EnumerationStats:
    executions: int = 0

class Enumeration:
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
            elif isinstance(ps, MapIterStart):
                map_results_weights = self.enumerate_map(
                    map_enter_ps=ps,
                    max_executions=max_executions,
                )
                for map_exit_ps, w in map_results_weights:
                    item = PrioritizedItem(-(cum_weight + w), map_exit_ps)
                    heapq.heappush(frontier, item)
            elif isinstance(ps, MapIterEnd):
                return_scores[ps.value] = logsumexp(
                    return_scores.get(ps.value, float('-inf')),
                    cum_weight
                )
                executions += 1
            else:
                raise ValueError("Unrecognized program state message")
        if self._stats is not None:
            self._stats.executions += executions
        return return_scores

    def enumerate_map(
        self,
        map_enter_ps: MapIterStart,
        max_executions: int,
    ) -> Sequence[Tuple[ProgramState, float]]:
        # TODO: add logic to allow for maintaining and reading from global store
        all_result_probs = []
        for i in map_enter_ps.iterator:
            ps_i = map_enter_ps.step(i)
            result_scores = self.enumerate_tree(ps_i, max_executions)
            all_result_probs.append(result_scores.items())
        finish_ps = map_enter_ps.map_finish_program_state
        map_result_weights = []
        for output_values in product(*all_result_probs):
            cum_weight = 0
            outputs = []
            for output, weight in output_values:
                cum_weight += weight
                outputs.append(output)
            map_exit_ps = finish_ps.step(outputs)
            map_result_weights.append((map_exit_ps, cum_weight))
        return map_result_weights

    def run(self, *args, **kws) -> Categorical:
        init_ps = CPSInterpreter().initial_program_state(self.function)
        ps = init_ps.step(*args, **kws)
        return_scores = self.enumerate_tree(ps, self.max_executions)
        return Categorical.from_dict(softmax_dict(return_scores))

    def _run_with_stats(self, *args, **kws) -> Tuple[Categorical, EnumerationStats]:
        self._stats = EnumerationStats()
        result = self.run(*args, **kws)
        self._stats, stats = None, self._stats
        return result, stats

