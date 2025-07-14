import heapq
import linecache
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Union, List, Tuple, Dict, Set, Callable, Generic
from functools import cached_property

from flippy.core import ProgramState, ReturnState, SampleState, ObserveState, InitialState, GlobalStore
from flippy.interpreter import CPSInterpreter
from flippy.types import Element
from flippy.transforms import CPSTransform
from flippy.distributions import Categorical, ZeroDistributionError
from flippy.tools import logsumexp
from flippy.callentryexit import EnterCallState, ExitCallState
from flippy.map import MapEnter, MapExit
from flippy.inference.simpleenumeration import EnumerationStats, ProgramStateRecord
from flippy.inference.inference import InferenceAlgorithm, DiscreteInferenceResult
from flippy.hashable import hashabledict
from flippy.tools import LRUCache
from flippy.distributions.support import ClosedInterval

try:
    from joblib import Parallel, delayed, cpu_count
except ImportError:
    pass

@dataclass
class ScoredProgramState:
    cumulative_score: float
    program_state: tuple

    def __lt__(self, other: 'ScoredProgramState'):
        return self.cumulative_score > other.cumulative_score

    def __eq__(self, other: 'ScoredProgramState'):
        return self.cumulative_score == other.cumulative_score

    def __gt__(self, other: 'ScoredProgramState'):
        return self.cumulative_score < other.cumulative_score

    def __iter__(self):
        return iter((self.cumulative_score, self.program_state))

@dataclass
class ExecutionResult:
    states: List[ProgramState]
    scores: List[float]
    cont_var_assigned: bool = False

    def collapse(self):
        if len(self) == 0:
            return self
        assert len(self.states) == len(self.scores)
        states_scores = defaultdict(lambda : float('-inf'))
        for state, score in zip(self.states, self.scores):
            states_scores[state] = logsumexp(states_scores[state], score)
        states, scores = zip(*states_scores.items())
        return ExecutionResult(states, scores, **self.info)

    def add_score(self, score):
        new_scores = [s + score for s in self.scores]
        return ExecutionResult(self.states, new_scores, **self.info)

    @property
    def info(self) -> Dict[str, Any]:
        return dict(cont_var_assigned=self.cont_var_assigned)

    def __iter__(self):
        yield from zip(self.states, self.scores)

    def __len__(self):
        assert len(self.states) == len(self.scores)
        return len(self.states)

@dataclass
class ReturnExecutionBranch(ExecutionResult):
    states: List[Union[ReturnState, ExitCallState]]
    scores: List[float]

class Enumeration(InferenceAlgorithm,Generic[Element]):
    def __init__(
        self,
        function,
        max_states=float('inf'),
        _call_cache_size=128,
        _map_cross_product=True,
        _cpus=1,
        _emit_call_entryexit=True,
        _state_visit_callback: Callable[[ProgramState], None]=None,
        _track_sample_state_visits=False,
        cont_var_func : Callable[[SampleState], List] = None
    ):
        assert _cpus > 0, "Number of CPUs must be greater than 0"
        self.cps = CPSInterpreter(_emit_call_entryexit=_emit_call_entryexit)
        if not CPSTransform.is_transformed(function):
            function = self.cps.non_cps_callable_to_cps_callable(function)
        self.function = function
        self.max_states = max_states
        self._max_states_per_partition = max_states // _cpus if max_states < float('inf') else float('inf')
        self._track_sample_state_visits = _track_sample_state_visits or (max_states < float('inf'))
        if _call_cache_size > 0:
            self._call_cache = LRUCache(max_size=_call_cache_size)
        else:
            self._call_cache = None
        self._map_cross_product = _map_cross_product
        self._cpus = _cpus
        self._emit_call_entryexit = _emit_call_entryexit
        self._cont_var_func = cont_var_func
        self._n_sample_states_visited = None
        self._state_visit_callback = self._create_state_visit_callback(_state_visit_callback)


    def _create_state_visit_callback(
        self,
        _state_visit_callback: Callable[[ProgramState], None]=None
    ):
        if self._track_sample_state_visits:
            def increment_state_visit_count(ps):
                if isinstance(ps, SampleState):
                    self._n_sample_states_visited += 1
                if _state_visit_callback is not None:
                    _state_visit_callback(ps)
            return increment_state_visit_count


    @cached_property
    def init_ps(self):
        return self.cps.initial_program_state(self.function)


    def set_cont_var_func(self, f: Callable[[SampleState], List]):
        self._cont_var_func = f


    def run(self, *args, **kws) -> DiscreteInferenceResult[Element]:
        return_values, return_scores = self._run(*args, **kws)
        return DiscreteInferenceResult.from_values_scores(return_values, return_scores)


    def _run(self, *args, **kws):
        if self._track_sample_state_visits:
            self._n_sample_states_visited = 0
        if self._cpus == 1:
            return_values, return_scores = self._run_single(*args, **kws)
        else:
            return_values, return_scores = self._run_parallel(*args, **kws)
        return return_values, return_scores


    def _run_single(self, *args, **kws):
        ps, score = self.next_choice_state(self.init_ps, args=args, kwargs=kws)
        result = self.enumerate_return_states_scores(ps)
        if len(result) == 0:
            raise ZeroDistributionError
        result = result.add_score(score)
        return_vals = [rs.value for rs in result.states]
        return return_vals, result.scores


    def _run_parallel(self, *args, **kws):
        assert CPSTransform.is_transformed(self.function), \
            "Function must be CPS transformed prior to creating workers"
        if self._cpus < 0:
            cpus = cpu_count() + 1 + self._cpus
        else:
            cpus = self._cpus

        all_value_scores = Parallel(n_jobs=cpus, backend="loky")(
            delayed(self._run_partition)(
                *args, **kws,
                _partition_idx=i,
                _partitions=cpus,
                _linecache=linecache.cache,
            )
        for i in range(cpus))
        values = sum([s for s, _ in all_value_scores], [])
        scores = sum([s for _, s in all_value_scores], [])
        return values, scores


    def _run_partition(self, *args, _partition_idx, _partitions, _linecache, **kws):
        linecache.cache = _linecache # restore linecache so inspect.getsource works
        ps, init_score = self.next_choice_state(self.init_ps, args=args, kwargs=kws)
        result = self.enumerate_return_states_scores(
            init_ps=ps,
            _partition_idx=_partition_idx,
            _partitions=_partitions
        )
        result = result.add_score(init_score)
        values = [state.value for state in result.states]
        return values, result.scores


    def enumerate_return_states_scores(
        self,
        init_ps: ProgramState,
        _partition_idx=0,
        _partitions=1
    ) -> ReturnExecutionBranch:
        assert _partition_idx < _partitions, "Partition index must be less than the number of partitions"
        if isinstance(init_ps, (ReturnState, ExitCallState)):
            return ReturnExecutionBranch([init_ps], [0.])
        frontier: List[ScoredProgramState] = []
        return_states = []
        return_scores = []
        heapq.heappush(frontier, ScoredProgramState(0., init_ps))
        in_partition = _partitions == 1
        result = None
        while len(frontier) > 0:
            if not in_partition and len(frontier) >= _partitions:
                # This block of code only occurs once in each worker during
                # multi-cpu enumeration (and never in single-cpu enumeration).
                # We partition trace space by expanding the frontier until there
                # are at least as many program states as there are partitions.
                # Then we assign a subset of the frontier to each worker.
                if _partition_idx != 0:
                    return_states, return_scores = [], []
                sub_frontier = []
                for i, ps in enumerate(frontier):
                    if i % _partitions == _partition_idx:
                        sub_frontier.append(ps)
                frontier = sub_frontier
                in_partition = True
                continue

            if (
                self._track_sample_state_visits and \
                self._n_sample_states_visited >= self._max_states_per_partition
            ):
                return ReturnExecutionBranch(return_states, return_scores)

            cum_score, ps = heapq.heappop(frontier)
            result = self.enumerate_successors_scores(ps)

            for new_ps, score in result:
                if score == float('-inf'):
                    continue
                if isinstance(new_ps, (ReturnState, ExitCallState)):
                    return_states.append(new_ps)
                    return_scores.append(cum_score + score)
                else:
                    heapq.heappush(frontier, ScoredProgramState(cum_score + score, new_ps))

        return ReturnExecutionBranch(return_states, return_scores, **result.info)


    def enumerate_successors_scores(
        self,
        ps: ProgramState,
    ) -> ExecutionResult:
        # we enumerate successors of a program state differently depending on
        # what kind of state it is
        if isinstance(ps, SampleState):
            return self.enumerate_sample_state_successors(ps)
        elif isinstance(ps, EnterCallState):
            result = self.enumerate_enter_call_state_successors(ps)
            # we need to take the next step for each successor
            successors = []
            scores = []
            for exit_state, exit_score in result:
                new_ps, new_score = self.next_choice_state(exit_state)
                if new_score == float('-inf'):
                    continue
                successors.append(new_ps)
                scores.append(exit_score + new_score)
            return ExecutionResult(successors, scores, **result.info)
        elif isinstance(ps, MapEnter):
            raise NotImplementedError
        elif isinstance(ps, InitialState):
            new_ps, step_score = self.next_choice_state(ps)
            if step_score > float('-inf'):
                successors = [new_ps]
                scores = [step_score]
            else:
                successors = []
                scores = []
            return ExecutionResult(successors, scores)
        else:
            raise ValueError(f"Unrecognized program state message {ps}")


    def enumerate_sample_state_successors(
        self,
        ps: SampleState,
    ) -> ExecutionResult:
        if isinstance(ps.distribution.support, ClosedInterval):
            assert self._cont_var_func is not None, \
                "Continuous sample state values must be provided for continuous distributions"
            values = self._cont_var_func(ps)
            cont_var_assigned = True
        else:
            assert not ps.fit, f"Enumeration doesn't support Distribution.fit: {ps.name}"
            values = ps.distribution.support
            cont_var_assigned = False
        successors, scores = [], []
        for value in values:
            score = ps.distribution.log_probability(value)
            new_ps, new_score = self.next_choice_state(ps, value=value)
            if new_score == float('-inf'):
                continue
            score += new_score
            successors.append(new_ps)
            scores.append(score)
        return ExecutionResult(successors, scores, cont_var_assigned=cont_var_assigned)

    def enumerate_enter_call_state_successors(
        self,
        enter_state: EnterCallState,
    ) -> ExecutionResult:
        if self._call_cache is None:
            return self._enumerate_enter_call_state_successors(enter_state)

        # we never cache the root call
        is_root_call = len(enter_state.stack) == 1
        if is_root_call:
            return self._enumerate_enter_call_state_successors(enter_state)

        # This logic handles caching using an LRU cache
        global_store_key = hashabledict(enter_state.init_global_store.store)
        key = (enter_state.function, enter_state.args, enter_state.kwargs, global_store_key)
        if key in self._call_cache:
            exit_values, exit_scores = self._call_cache[key]
            exit_states = []
            for rv, gs in exit_values:
                gs: GlobalStore
                next_state = enter_state.skip(rv)
                next_state.set_init_global_store(gs.copy(), force=True)
                exit_states.append(next_state)
            result = ExecutionResult(exit_states, exit_scores)
            if self._track_sample_state_visits:
                self._n_sample_states_visited += len(exit_states) > 1
        else:
            result = self._enumerate_enter_call_state_successors(enter_state)
            # we can only cache if no continuous variables were assigned during the call
            if not result.cont_var_assigned:
                exit_values = [(rs.value, rs.init_global_store) for rs in result.states]
                self._call_cache[key] = (exit_values, result.scores)
        return result

    def _enumerate_enter_call_state_successors(
        self,
        init_ps: EnterCallState
    ) -> ExecutionResult:
        # when we enter a call, we take the first step, see if it halts or returns immediately
        assert isinstance(init_ps, EnterCallState)
        ps, init_score = self.next_choice_state(init_ps)
        if init_score == float('-inf'):
            return ExecutionResult([], [])
        if isinstance(ps, ExitCallState):
            return ExecutionResult([ps], [init_score])

        # if it doesn't immediately exit, we enumerate to get all the exit states/scores
        result = self.enumerate_return_states_scores(init_ps=ps)
        result = result.collapse()
        result = result.add_score(init_score)
        return result


    def next_choice_state(
        self,
        init_ps: ProgramState,
        *,
        value=None,
        args=None,
        kwargs=None,
    ) -> Tuple[ProgramState, float]:
        """
        This runs a program state until it reaches a choice state or an exit
        state. It returns the next choice state and score collected along the way.
        """

        if isinstance(init_ps, ObserveState):
            score = init_ps.distribution.log_probability(init_ps.value)
        else:
            score = 0.
        if isinstance(init_ps, (SampleState, MapExit)):
            ps = init_ps.step(value)
        elif isinstance(init_ps, InitialState):
            ps = init_ps.step(*args, **kwargs)
            # We skip the call entry state associated with the root call
            # since (1) we don't want to cache it and (2) don't want to consider it
            # for partitioning across workers
            if isinstance(ps, EnterCallState) and ps.is_root_call:
                ps = ps.step()
        else:
            ps = init_ps.step()
        while not isinstance(ps, (SampleState, EnterCallState, ReturnState, ExitCallState)):
            if self._state_visit_callback:
                self._state_visit_callback(ps)
            if isinstance(ps, ObserveState):
                score += ps.distribution.log_probability(ps.value)
            if score == float('-inf'):
                return None, float('-inf')
            ps = ps.step()
        if self._state_visit_callback:
            self._state_visit_callback(ps)
        return ps, score

