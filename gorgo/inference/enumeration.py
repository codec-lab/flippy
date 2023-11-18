import heapq
import math
from collections import defaultdict
import dataclasses
from typing import Any, Union, List

from gorgo.core import ProgramState, ReturnState, SampleState, ObserveState, InitialState
from gorgo.interpreter import CPSInterpreter
from gorgo.distributions import Categorical

@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=dataclasses.field(compare=False)

class Enumeration:
    def __init__(self, function, max_executions=float('inf')):
        self.function = function
        self.max_executions = max_executions

    def enumerate_tree(
        self,
        ps: ProgramState,
        max_executions: int,
    ):
        frontier: List[PrioritizedItem] = []
        return_probs = defaultdict(float)
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
                cum_prob = math.exp(cum_weight)
                assert cum_prob > 0, "Possible underflow"
                return_probs[ps.value] += cum_prob
                executions += 1
            else:
                raise ValueError("Unrecognized program state message")
        return return_probs

    def run(self, *args, **kws):
        init_ps = CPSInterpreter().initial_program_state(self.function)
        ps = init_ps.step(*args, **kws)
        return_probs = self.enumerate_tree(ps, self.max_executions)
        total_prob = sum(return_probs.values())
        return_probs = {e: p/total_prob for e, p in return_probs.items()}
        return Categorical.from_dict(return_probs)
