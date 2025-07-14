
from scipy.optimize import minimize
import numpy as np
from typing import Tuple
import linecache

from joblib import Parallel, delayed

from flippy.core import ReturnState, SampleState, ObserveState, ProgramState
from flippy.distributions.random import default_rng, RandomNumberGenerator
from flippy.distributions import ZeroDistributionError
from flippy.interpreter import CPSInterpreter
from flippy.distributions import Categorical, RandomNumberGenerator
from flippy.distributions.support import ClosedInterval
from flippy.inference import Enumeration
from flippy.types import Element
from flippy.inference.inference import InferenceAlgorithm, MarginalLikelihood, \
    DiscreteInferenceResult

class MaximumMarginalAPosteriori(InferenceAlgorithm[Element]):
    def __init__(
        self, function, seed=None, method="Powell",
        disp=False, maxiter=1000, maxfev=1000,
        maximum_likelihood=False,
        min_valid_runs=1,
        max_retries=10,
        _cpus=1,
        **enumeration_kwargs
    ):
        self.function = function
        self.enumeration_alg = Enumeration(
            function,
            **enumeration_kwargs
        )
        self.seed = seed
        self.method = method
        self.maxiter = maxiter
        self.maxfev = maxfev
        self.disp = disp
        self.maximum_likelihood = maximum_likelihood
        self.min_valid_runs = min_valid_runs
        self.max_retries = max_retries
        self._cpus = _cpus

    def single_run(self, *args, **kwargs) -> Tuple[MarginalLikelihood, DiscreteInferenceResult[Element]]:
        rng = RandomNumberGenerator(self.seed)
        done = False
        assignment = {}
        while not done:
            assignment, done = self.optimize_with_assignment(args, kwargs, assignment, rng=rng)
        _, score, result = self.assignment_score(args, kwargs, assignment, rng=rng)
        if result is None:
            raise ZeroDistributionError
        return score, result

    def _run(self, *args, **kwargs) -> Tuple[float, DiscreteInferenceResult[Element]]:
        def job(linecache_cache) -> Tuple[float, DiscreteInferenceResult[Element]]:
            linecache.cache.update(linecache_cache)
            try:
                return self.single_run(*args, **kwargs)
            except ZeroDistributionError:
                return float('-inf'), None
        assert self._cpus > 0
        with Parallel(n_jobs=self._cpus) as parallel:
            valid_results = []
            retries = 0
            while len(valid_results) < self.min_valid_runs:
                results = parallel(
                    delayed(job)(linecache.cache) for _ in range(self._cpus)
                )
                retries += len(results)
                valid_results.extend([r for r in results if r[0] > float('-inf')])
                if retries >= self.max_retries:
                    break
        if len(valid_results) == 0:
            raise ZeroDistributionError("Unable to find minimum number of valid solutions.")
        best_score, best_result = max(valid_results, key=lambda x: x[0])
        metadata = dict(retries=retries, n_valid_results=len(valid_results))
        return best_score, best_result, metadata

    def run(self, *args, **kwargs) -> 'MaximumMarginalAPosterioriResult[Element]':
        best_score, best_result, metadata = self._run(*args, **kwargs)
        values, probs = zip(*best_result.items())
        result = MaximumMarginalAPosterioriResult(values, probs, np.exp(best_score))
        result.update_metadata(metadata)
        return result

    def optimize_with_assignment(self, args, kwargs, assignment: dict, rng=default_rng):
        init_assignments, _, _ = self.assignment_score(args, kwargs, assignment, rng=rng)
        varnames = list(init_assignments.keys())
        init_values = [init_assignments[v][0] for v in varnames]
        bounds = [init_assignments[v][1] for v in varnames]
        def objective(x):
            assignment = dict(zip(varnames, zip(x, bounds)))
            new_assignment, log_marginal_likelihood, _ = \
                self.assignment_score(args, kwargs, assignment, rng=rng)
            if len(new_assignment) > len(assignment):
                raise NewVariableException(new_assignment)
            return -np.exp(log_marginal_likelihood)
        try:
            res = minimize(
                objective,
                init_values,
                method=self.method,
                options={
                    "disp": self.disp,
                    "maxiter": self.maxiter,
                    "maxfev": self.maxfev
                },
                bounds=bounds
            )
            new_assignment = dict(zip(varnames, zip(res.x, bounds)))
            done = res.success
        except NewVariableException as res:
            new_assignment = res.args[0]
            done = False
        return new_assignment, done

    def assignment_score(self, args, kwargs, assignments: dict, rng=default_rng):
        assert all(isinstance(v, tuple) for v in assignments.values())
        assignments = assignments.copy()
        cont_var_sample_score = [0]
        def cont_var_func(ps: SampleState):
            if ps.name not in assignments:
                assert isinstance(ps.distribution.support, ClosedInterval)
                if ps.initial_value in ps.distribution.support:
                    value = ps.initial_value
                else:
                    value = ps.distribution.sample(rng=rng)
                bounds = ps.distribution.support.start, ps.distribution.support.end
                assignments[ps.name] = (value, bounds)
            value, bounds = assignments[ps.name]
            cont_var_sample_score[0] += ps.distribution.log_probability(value)
            return [value]
        self.enumeration_alg.set_cont_var_func(cont_var_func)
        try:
            result = self.enumeration_alg.run(*args, **kwargs)
            score = np.log(result.marginal_likelihood) if result.marginal_likelihood > 0 else float("-inf")
        except ZeroDistributionError:
            result = None
            score = float("-inf")
        if self.maximum_likelihood:
            score -= cont_var_sample_score[0]
        return assignments, score, result

class NewVariableException(Exception):
    pass

class MaximumMarginalAPosterioriResult(DiscreteInferenceResult[Element]):
    retries : int
    n_valid_results : int

    def update_metadata(self, metadata: dict):
        self.retries = metadata.get("retries", None)
        self.n_valid_results = metadata.get("n_valid_results", None)
