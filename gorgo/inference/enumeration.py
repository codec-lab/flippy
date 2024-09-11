import queue
import heapq
import linecache
from dataclasses import dataclass
from itertools import product
from collections import defaultdict, OrderedDict
from typing import Any, Union, List, Tuple, Dict, Set, Callable
from functools import cached_property

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye as sp_eye, coo_array

from gorgo.core import ProgramState, ReturnState, SampleState, ObserveState, InitialState, GlobalStore
from gorgo.interpreter import CPSInterpreter
from gorgo.transforms import CPSTransform
from gorgo.distributions import Categorical
from gorgo.tools import logsumexp
from gorgo.callentryexit import EnterCallState, ExitCallState
from gorgo.map import MapEnter, MapExit
from gorgo.inference.simpleenumeration import EnumerationStats, ProgramStateRecord
from gorgo.hashable import hashabledict
from gorgo.tools import LRUCache

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


class Enumeration:
    def __init__(
        self,
        function,
        max_states=float('inf'),
        _call_cache_size=128,
        _map_cross_product=True,
        _enumeration_strategy='tree',
        _cpus=1,
        _emit_call_entryexit=True,
    ):
        self.cps = CPSInterpreter(_emit_call_entryexit=_emit_call_entryexit)
        if not CPSTransform.is_transformed(function):
            function = self.cps.non_cps_callable_to_cps_callable(function)
        self.function = function
        self.max_states = max_states
        self._stats = None
        if _call_cache_size > 0:
            self._call_cache = LRUCache(max_size=_call_cache_size)
        else:
            self._call_cache = None
        self._enumeration_strategy = _enumeration_strategy
        self._map_cross_product = _map_cross_product
        self._cpus = _cpus
        self._emit_call_entryexit = _emit_call_entryexit

    @cached_property
    def init_ps(self):
        return self.cps.initial_program_state(self.function)

    def _run_partition(self, *args, _partition_idx, _partitions, _linecache, **kws):
        # restore linecache so inspect.getsource works for interactively defined functions
        linecache.cache = _linecache

        ps = self.init_ps.step(*args, **kws)
        # This skips the initial call entry state when the outermost function is called
        # otherwise partitioning would only happen after all possible executions have been
        # fully enumerated
        # TODO: fixing partitioning scheme to handle recursive enumeration calls would
        # also deal with this
        if isinstance(ps, EnterCallState):
            assert len(ps.stack) == 1
            ps = ps.step()
        states, scores = self.enumerate_return_states_scores(
            init_ps=ps,
            max_states=self.max_states,
            _partition_idx=_partition_idx,
            _partitions=_partitions
        )
        values = [state.value for state in states]
        return values, scores

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

    def _run_single(self, *args, **kws):
        ps = self.init_ps.step(*args, **kws)
        return_states, return_scores = self.enumerate_return_states_scores(ps, self.max_states)
        return_vals = [rs.value for rs in return_states]
        return return_vals, return_scores

    def run(self, *args, **kws):
        if self._cpus == 1:
            return_values, return_scores = self._run_single(*args, **kws)
        else:
            return_values, return_scores = self._run_parallel(*args, **kws)
        return_probs = np.exp(return_scores)
        return_probs = return_probs / return_probs.sum()
        normalized_dist = {}
        for rv, rp in zip(return_values, return_probs):
            normalized_dist[rv] = normalized_dist.get(rv, 0.) + rp
        return Categorical.from_dict(normalized_dist)

    def enumerate_return_states_scores(
        self,
        init_ps: ProgramState,
        max_states: int = float('inf'),
        _partition_idx=0,
        _partitions=1
    ) -> Tuple[List[Union[ReturnState,ExitCallState]], List[float]]:
        assert _partition_idx < _partitions, "Partition index must be less than the number of partitions"
        if self._enumeration_strategy == 'graph':
            return self.enumerate_graph(init_ps, max_states, _partition_idx=_partition_idx, _partitions=_partitions)
        elif self._enumeration_strategy == 'tree':
            return self.enumerate_tree(init_ps, max_states, _partition_idx=_partition_idx, _partitions=_partitions)
        else:
            raise ValueError(f"Unrecognized enumeration strategy {self._enumeration_strategy}")


    def enumerate_tree(
        self,
        init_ps: ProgramState,
        max_states: int = float('inf'),
        _partition_idx=0,
        _partitions=1
    ) -> Dict[Union[ReturnState,ExitCallState], float]:
        if isinstance(init_ps, (ReturnState, ExitCallState)):
            return [init_ps], [0.]
        frontier: List[ScoredProgramState] = []
        n_visited = 0
        return_states = []
        return_scores = []
        heapq.heappush(frontier, ScoredProgramState(0., init_ps))
        in_partition = _partitions == 1
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

            if n_visited >= max_states:
                break
            cum_score, ps = heapq.heappop(frontier)
            successors, scores = self.enumerate_successors_scores(ps)

            for new_ps, score in zip(successors, scores):
                if score == float('-inf'):
                    continue
                if isinstance(new_ps, (ReturnState, ExitCallState)):
                    return_states.append(new_ps)
                    return_scores.append(cum_score + score)
                else:
                    heapq.heappush(frontier, ScoredProgramState(cum_score + score, new_ps))
            n_visited += 1
            if self._stats is not None:
                self._stats.states_visited.append(ProgramStateRecord(ps.__class__, ps.name))

        return return_states, return_scores


    def enumerate_graph(
        self,
        init_ps: ProgramState,
        max_states: int = float('inf'),
        _partition_idx=0,
        _partitions=1
    ) -> Dict[Union[ReturnState,ExitCallState], float]:
        if isinstance(init_ps, (ReturnState, ExitCallState)):
            return [init_ps], [0.]
        assert _partitions == 1, "Graph enumeration does not support partitioning"

        transition_scores: Dict[Tuple[ProgramState, ProgramState], float] = \
            defaultdict(lambda: float('-inf'))

        # Queue for BFS, which lets us get states in topological order (?)
        frontier: queue.Queue[SampleState] = queue.Queue()
        frontier.put(init_ps)
        visited : Set[SampleState] = set([init_ps])
        return_states: List[Union[ReturnState, ExitCallState]] = []
        state_idx = {init_ps: 0}
        while not frontier.empty():
            if len(visited) >= max_states:
                break
            ps = frontier.get()
            successors, scores = self.enumerate_successors_scores(ps)

            # logic for updating graph
            for new_ps, score in zip(successors, scores):
                if score == float('-inf'):
                    continue
                if new_ps not in visited and new_ps not in return_states:
                    if isinstance(new_ps, (ReturnState, ExitCallState)):
                        return_states.append(new_ps)
                    else:
                        state_idx[new_ps] = len(state_idx)
                        visited.add(new_ps)
                    if isinstance(new_ps, (SampleState, EnterCallState)):
                        frontier.put(new_ps)
                transition_scores[(ps, new_ps)] = \
                    logsumexp(transition_scores[(ps, new_ps)], score)

            if self._stats is not None:
                self._stats.states_visited.append(ProgramStateRecord(ps.__class__, ps.name))

        if len(return_states) == 0:
            return [], []

        for ps in return_states:
            state_idx[ps] = len(state_idx)
        n_states = len(visited) + len(return_states)

        # Sparse matrix implementation
        # construct probability/exp(score) matrix
        ps1_ps2, scores = zip(*transition_scores.items())
        ps1, ps2 = zip(*ps1_ps2)
        ps1 = [state_idx[ps] for ps in ps1]
        ps2 = [state_idx[ps] for ps in ps2]
        sp_scores_matrix = coo_array(
            (np.exp(scores), (ps1, ps2)),
            (n_states, n_states)
        ).tocsc()

        # construct unnormalized return probability vector
        sp_return_onehot = sp_eye(
            n_states, len(return_states), k=-(n_states - len(return_states))
        ).tocsc()
        eye = sp_eye(n_states).tocsc()
        sp_return_probs = spsolve(
            A=eye - sp_scores_matrix,
            b=sp_return_onehot
        )
        if sp_return_probs.ndim == 2:
            sp_return_logprobs = np.log(sp_return_probs[0].toarray().flatten())
        else:
            sp_return_logprobs = np.log([sp_return_probs[0]])
        return return_states, sp_return_logprobs

    def enumerate_successors_scores(
        self,
        ps: ProgramState,
    ) -> Tuple[List[ProgramState], List[float]]:
        # we enumerate successors of a program state differently depending on
        # what kind of state it is
        if isinstance(ps, SampleState):
            successors, scores = self.enumerate_sample_state_successors(ps)
        elif isinstance(ps, EnterCallState):
            exit_states, exit_scores = self.enumerate_enter_call_state_successors(ps)
            # we need to take the next step for each successor
            successors = []
            scores = []
            for exit_state, exit_score in zip(exit_states, exit_scores):
                new_ps, new_score = self.next_choice_state(exit_state)
                if new_score == float('-inf'):
                    continue
                successors.append(new_ps)
                scores.append(exit_score + new_score)
        elif isinstance(ps, MapEnter):
            if self._map_cross_product:
                successors, scores = self.enumerate_enter_map_state_successors(ps)
            else:
                successors = [ps.step(False)]
                scores = [0.]
        elif isinstance(ps, InitialState):
            new_ps, step_score = self.next_choice_state(ps)
            if step_score > float('-inf'):
                successors = [new_ps]
                scores = [step_score]
            else:
                successors = []
                scores = []
        else:
            raise ValueError(f"Unrecognized program state message {ps}")
        return successors, scores


    def next_choice_state(
        self,
        init_ps: ProgramState,
        value=None
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
        else:
            ps = init_ps.step()
        while not isinstance(ps, (SampleState, EnterCallState, ReturnState, ExitCallState)):
            if isinstance(ps, ObserveState):
                score += ps.distribution.log_probability(ps.value)
            if score == float('-inf'):
                return None, float('-inf')
            ps = ps.step()
            if self._stats is not None:
                self._stats.states_visited.append(ProgramStateRecord(ps.__class__, ps.name))
        return ps, score


    def enumerate_sample_state_successors(
        self,
        ps: SampleState,
    ) -> Tuple[List[ProgramState], List[float]]:
        successors, scores = [], []
        for value in ps.distribution.support:
            score = ps.distribution.log_probability(value)
            new_ps, new_score = self.next_choice_state(ps, value)
            if new_score == float('-inf'):
                continue
            score += new_score
            successors.append(new_ps)
            scores.append(score)
        return successors, scores


    def enumerate_enter_call_state_successors(
        self,
        enter_state: EnterCallState,
    ) -> Tuple[List[ExitCallState], List[float]]:
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
        else:
            exit_states, exit_scores = self._enumerate_enter_call_state_successors(enter_state)
            exit_values = [(rs.value, rs.init_global_store) for rs in exit_states]
            self._call_cache[key] = (exit_values, exit_scores)
        return exit_states, exit_scores

    def _enumerate_enter_call_state_successors(
        self,
        init_ps: EnterCallState
    ) -> Tuple[List[ExitCallState], List[float]]:
        # when we enter a call, we take the first step, see if it halts or returns immediately
        assert isinstance(init_ps, EnterCallState)
        ps, init_score = self.next_choice_state(init_ps)
        if init_score == float('-inf'):
            return [], []
        if isinstance(ps, ExitCallState):
            return [ps], [init_score]

        # if it doesn't immediately exit, we enumerate to get all the exit states/scores
        successors, scores = self.enumerate_return_states_scores(init_ps=ps, max_states=self.max_states)
        scores = [init_score + score for score in scores]

        # if we're using a tree enumeration strategy, we need to collapse over exit states
        # note this is similar to return site caching
        if self._enumeration_strategy == 'tree' and len(successors) > 0:
            successor_scores = defaultdict(lambda : float('-inf'))
            for state, score in zip(successors, scores):
                successor_scores[state] = logsumexp(successor_scores[state], score)
            successors, scores = zip(*successor_scores.items())
        return successors, scores

    def enumerate_enter_map_state_successors(
        self,
        map_enter_ps : MapEnter
    ):
        ps : EnterCallState = map_enter_ps.step(True)
        init_node = MapCrossProductNode(ps)
        successors, scores = init_node.enumerate_exit_map_state_successors(self)
        return successors, scores

    def _run_with_stats(self, *args, **kws) -> Tuple[Categorical, EnumerationStats]:
        self._stats = EnumerationStats()
        result = self.run(*args, **kws)
        self._stats, stats = None, self._stats
        return result, stats

@dataclass
class MapCrossProductNode:
    """
    Helper class for keeping track of mapped values and scores.
    This is needed to handle changes to the global store.
    """
    enter_call_ps: Union[EnterCallState, MapExit]
    parent_exit_scores: List[Tuple[Any,float]] = None
    children: List['MapCrossProductNode'] = None

    def enumerate_exit_map_state_successors(self, enumerator: Enumeration):
        self.iteratively_expand(enumerator=enumerator)
        map_successors = []
        map_scores = []
        for iteration_results, ps in self.traverse():
            for vals_scores in product(*iteration_results):
                vals, scores = zip(*vals_scores)
                new_ps, new_score = enumerator.next_choice_state(ps, vals)
                score = sum(scores) + new_score
                if score == float('-inf'):
                    continue
                map_successors.append(new_ps)
                map_scores.append(score)
        return map_successors, map_scores

    def iteratively_expand(self, enumerator: Enumeration):
        frontier: List[MapCrossProductNode] = [self]
        while frontier:
            node = frontier.pop()
            if node.iteration_terminated():
                continue
            node.expand(enumerator)
            frontier.extend(node.children)

    def expand(self, enumerator: Enumeration):
        assert self.children is None, "Node already expanded"

        # enumerate values and scores for exiting the call
        # we want to partition them by the next state
        # (i.e., the subsequent global scope)
        exit_call_ps_list, exit_scores = \
            enumerator.enumerate_enter_call_state_successors(self.enter_call_ps)
        next_state_exit_val_scores = defaultdict(list)
        for exit_ps, score in zip(exit_call_ps_list, exit_scores):
            next_ps = exit_ps.step()
            next_state_exit_val_scores[next_ps].append((exit_ps.value, score))

        # create child nodes
        self.children = []
        for next_ps, exit_val_scores in next_state_exit_val_scores.items():
            self.children.append(MapCrossProductNode(next_ps, parent_exit_scores=exit_val_scores))

    def iteration_terminated(self):
        return isinstance(self.enter_call_ps, MapExit)

    def traverse(self):
        assert self.children is not None, "Node not expanded"
        assert self.parent_exit_scores is None, "We should only traverse from the root node"
        frontier: List[Tuple['MapCrossProductNode', ...]] = [(self, )]
        while frontier:
            seq = frontier.pop()
            for node in seq[-1].children:
                if node.iteration_terminated():
                    response_cross_prod = [
                        n.parent_exit_scores for n in seq[1:] + (node, )
                    ]
                    yield response_cross_prod, node.enter_call_ps
                else:
                    frontier.append(seq + (node, ))
