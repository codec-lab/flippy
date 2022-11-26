import random
from gorgo.core import StartingMessage, SampleMessage, ObserveMessage
from gorgo.interpreter import CPSInterpreter

class SamplePrior:
    def __init__(self, function, seed=None):
        self.function = function
        self.seed = seed

    def run(self, *args, **kws):
        rng = random.Random(self.seed)
        ps = CPSInterpreter().initial_program_state(self.function)
        trajectory = [ps]
        while not ps.is_returned():
            if isinstance(ps.message, StartingMessage):
                ps = ps.step(*args, **kws)
            elif isinstance(ps.message, SampleMessage):
                value = ps.message.distribution.sample(rng=rng)
                ps = ps.step(value)
            elif isinstance(ps.message, ObserveMessage):
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

from gorgo.core import StartingMessage, SampleMessage, ObserveMessage, ReturnMessage
from gorgo.interpreter import CPSInterpreter

@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=dataclasses.field(compare=False)

class Enumeration:
    def __init__(self, function):
        self.function = function
    def run(self, *args, **kws):
        frontier = []
        return_probs = defaultdict(float)
        ps = CPSInterpreter().initial_program_state(self.function)
        heapq.heappush(frontier, PrioritizedItem(0, ps))
        while len(frontier) > 0:
            item = heapq.heappop(frontier)
            cum_weight = -item.priority
            ps = item.item
            if isinstance(ps.message, StartingMessage):
                new_ps = ps.step(*args, **kws)
                heapq.heappush(frontier, PrioritizedItem(-cum_weight, new_ps))
            elif isinstance(ps.message, SampleMessage):
                for value in ps.message.distribution.support:
                    weight = ps.message.distribution.log_probability(value)
                    new_ps = ps.step(value)
                    heapq.heappush(frontier, PrioritizedItem(-(cum_weight + weight), new_ps))
            elif isinstance(ps.message, ObserveMessage):
                value = ps.message.value
                weight = ps.message.distribution.log_probability(value)
                if weight > float('-inf'):
                    new_ps = ps.step()
                    heapq.heappush(frontier, PrioritizedItem(-(cum_weight + weight), new_ps))
            elif isinstance(ps.message, ReturnMessage):
                cum_prob = math.exp(cum_weight)
                assert cum_prob > 0, "Possible underflow"
                return_probs[ps.message.value] += cum_prob
            else:
                raise ValueError("Unrecognized program state message")
        total_prob = sum(return_probs.values())
        return_probs = {e: p/total_prob for e, p in return_probs.items()}
        return return_probs