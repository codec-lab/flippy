import inspect
import linecache
import math
from collections import defaultdict
from typing import Generic

from flippy.core import ReturnState, SampleState, ObserveState
from flippy.types import Element
from flippy.interpreter import CPSInterpreter
from flippy.distributions import Categorical, RandomNumberGenerator
from flippy.inference.inference import InferenceAlgorithm


class LikelihoodWeighting(InferenceAlgorithm[Element]):
    def __init__(
        self,
        function,
        samples : int,
        seed=None,
        _cpus=1,
        _joblib_backend='loky'
    ):
        self.function = function
        self.samples = samples
        self.seed = seed
        self._cpus = _cpus
        self._joblib_backend = _joblib_backend

    def run(self, *args, **kws) -> Categorical[Element]:
        if self._cpus == 1:
            return_counts = self._run_batch(
                *args, **kws,
                samples=self.samples,
                seed=self.seed,
                _linecache=linecache.cache
            )
        else:
            return_counts = self._run_parallel(*args, **kws)
        total_prob = sum(return_counts.values())
        return_probs = {e: p/total_prob for e, p in return_counts.items()}
        return Categorical.from_dict(return_probs)

    def _run_batch(self, *args, samples: int, seed: int, _linecache, **kws):
        # restore linecache so inspect.getsource works for interactively defined functions
        linecache.cache = _linecache

        rng = RandomNumberGenerator(seed)
        init_ps = CPSInterpreter().initial_program_state(self.function)
        return_counts = defaultdict(float)
        for _ in range(samples):
            weight = 0
            ps = init_ps.step(*args, **kws)
            while not isinstance(ps, ReturnState):
                if isinstance(ps, SampleState):
                    assert not ps.fit, f"LikelihoodWeighting doesn't support Distribution.fit: {ps.name}"
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
        return return_counts

    def _run_parallel(self, *args, **kws):
        from joblib import Parallel, delayed, cpu_count
        rng = RandomNumberGenerator(self.seed)
        if self._cpus == -1:
            cpus = cpu_count()
        else:
            cpus = self._cpus
        all_return_counts = Parallel(n_jobs=cpus, backend=self._joblib_backend)(
            delayed(self._run_batch)(
                *args, **kws,
                samples=self.samples // cpus,
                seed=rng.new_seed(),
                _linecache=linecache.cache
            )
        for _ in range(self._cpus))
        return_counts = self._run_batch(
            *args, **kws,
            samples=self.samples % cpus,
            seed=rng.new_seed(),
            _linecache=linecache.cache,
        )
        for rc in all_return_counts:
            for k, v in rc.items():
                return_counts[k] += v
        return return_counts
