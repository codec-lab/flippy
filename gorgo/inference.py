import random
from gorgo.core import ProgramState, ReturnState, SampleState, ObserveState, InitialState
from gorgo.interpreter import CPSInterpreter

class SamplePrior:
    def __init__(self, function, seed=None):
        self.function = function
        self.seed = seed

    def run(self, *args, **kws):
        rng = random.Random(self.seed)
        ps = CPSInterpreter().initial_program_state(self.function)
        trajectory = [ps]
        while not isinstance(ps, ReturnState):
            if isinstance(ps, InitialState):
                ps = ps.step(*args, **kws)
            elif isinstance(ps, SampleState):
                value = ps.distribution.sample(rng=rng)
                ps = ps.step(value)
            elif isinstance(ps, ObserveState):
                ps = ps.step()
            else:
                raise ValueError("Unrecognized program state message")
            trajectory.append(ps)
        return trajectory

import heapq
import math
from collections import defaultdict
import dataclasses
from typing import Any

from gorgo.core import ProgramState, ReturnState, SampleState, ObserveState, InitialState
from gorgo.interpreter import CPSInterpreter

@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=dataclasses.field(compare=False)

class Enumeration:
    def __init__(self, function, max_executions=float('inf')):
        self.function = function
        self.max_executions = max_executions
    def run(self, *args, **kws):
        frontier = []
        return_probs = defaultdict(float)
        executions = 0
        ps = CPSInterpreter().initial_program_state(self.function)
        ps : ProgramState
        heapq.heappush(frontier, PrioritizedItem(0, ps))
        while len(frontier) > 0:
            if executions >= self.max_executions:
                break
            item = heapq.heappop(frontier)
            cum_weight = -item.priority
            ps = item.item
            if isinstance(ps, InitialState):
                new_ps = ps.step(*args, **kws)
                heapq.heappush(frontier, PrioritizedItem(-cum_weight, new_ps))
            elif isinstance(ps, SampleState):
                for value in ps.distribution.support:
                    weight = ps.distribution.log_probability(value)
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
        total_prob = sum(return_probs.values())
        return_probs = {e: p/total_prob for e, p in return_probs.items()}
        return return_probs