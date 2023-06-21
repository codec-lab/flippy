import random
import math
from collections import defaultdict

from gorgo.core import ReturnState, SampleState, ObserveState
from gorgo.interpreter import CPSInterpreter
from gorgo.distributions import Categorical, RandomNumberGenerator

class LikelihoodWeighting:
    def __init__(self, function, samples : int, seed=None):
        self.function = function
        self.samples = samples
        self.seed= seed

    def run(self, *args, **kws):
        rng = RandomNumberGenerator(self.seed)
        return_counts = defaultdict(float)
        init_ps = CPSInterpreter().initial_program_state(self.function)
        for _ in range(self.samples):
            weight = 0
            ps = init_ps.step(*args, **kws)
            while not isinstance(ps, ReturnState):
                if isinstance(ps, SampleState):
                    value = ps.distribution.sample(rng=rng)
                    ps = ps.step(value)
                elif isinstance(ps, ObserveState):
                    weight += ps.distribution.log_probability(ps.value)
                    if weight == float('-inf'):
                        break
                    ps = ps.step()
                else:
                    raise ValueError("Unrecognized program state")
            if weight == float('-inf'):
                continue

            return_counts[ps.value] += math.exp(weight)
        total_prob = sum(return_counts.values())
        return_probs = {e: p/total_prob for e, p in return_counts.items()}
        return Categorical.from_dict(return_probs)
