from collections import defaultdict
from typing import Generic
from flippy.core import ProgramState, ReturnState, SampleState, ObserveState, InitialState
from flippy.interpreter import CPSInterpreter
from flippy.distributions import Categorical, RandomNumberGenerator
from flippy.types import Element
from flippy.inference.inference import InferenceAlgorithm

class SamplePrior(InferenceAlgorithm[Element]):
    """Sample from the prior and ignore observation statements"""
    def __init__(self, function, samples : int, seed=None):
        self.function = function
        self.seed = seed
        self.samples = samples

    def run(self, *args, **kws) -> Categorical[Element]:
        rng = RandomNumberGenerator(self.seed)
        return_counts = defaultdict(int)
        init_ps = CPSInterpreter().initial_program_state(self.function)
        for _ in range(self.samples):
            ps = init_ps.step(*args, **kws)
            while not isinstance(ps, ReturnState):
                if isinstance(ps, SampleState):
                    value = ps.distribution.sample(rng=rng)
                    ps = ps.step(value)
                elif isinstance(ps, ObserveState):
                    ps = ps.step()
                else:
                    raise ValueError("Unrecognized program state message")
            return_counts[ps.value] += 1
        total_prob = sum(return_counts.values())
        return_probs = {e: p/total_prob for e, p in return_counts.items()}
        return Categorical.from_dict(return_probs)
