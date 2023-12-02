import heapq
import queue
import math
from collections import defaultdict, Counter
import dataclasses
from typing import Any, Union, List, Tuple, Dict, Set

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye as sp_eye, coo_array

from gorgo.core import ProgramState, ReturnState, SampleState, ObserveState, InitialState
from gorgo.interpreter import CPSInterpreter
from gorgo.distributions import Categorical
from gorgo.tools import logsumexp, softmax_dict
from gorgo.types import ReturnValue
from gorgo.callentryexit import EnterCallState, ExitCallState

@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=dataclasses.field(compare=False)

@dataclasses.dataclass(frozen=True)
class ProgramStateRecord:
    kind: type
    name: str

@dataclasses.dataclass
class EnumerationStats:
    states_visited: list[ProgramStateRecord] = dataclasses.field(default_factory=list)

    def site_counts(self):
        return Counter(self.states_visited)

class Enumeration:
    def __init__(self, function, max_executions=float('inf')):
        self.function = function
        self.max_executions = max_executions
        self._stats = None

    def enumerate_tree(
        self,
        ps: ProgramState,
        max_executions: int,
    ):
        frontier: List[PrioritizedItem] = []
        return_scores = {}
        # return_probs = defaultdict(float)
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
            elif isinstance(ps, (EnterCallState, ExitCallState)):
                new_ps = ps.step()
                heapq.heappush(frontier, PrioritizedItem(-cum_weight, new_ps))
            else:
                raise ValueError(f"Unrecognized program state message, {ps}")
            if self._stats is not None:
                self._stats.states_visited.append(ProgramStateRecord(ps.__class__, ps.name))
        return return_scores

    def run(self, *args, **kws):
        init_ps = CPSInterpreter().initial_program_state(self.function)
        ps = init_ps.step(*args, **kws)
        return_scores = self.enumerate_tree(ps, self.max_executions)
        return Categorical.from_dict(softmax_dict(return_scores))

    def _run_with_stats(self, *args, **kws) -> Tuple[Categorical, EnumerationStats]:
        self._stats = EnumerationStats()
        result = self.run(*args, **kws)
        self._stats, stats = None, self._stats
        return result, stats


class GraphEnumeration:
    def __init__(self, function, max_states=float('inf')):
        self.function = function
        self.max_states = max_states
        self._stats = None

    def run(self, *args, **kws):
        init_ps = CPSInterpreter().initial_program_state(self.function)
        ps = init_ps.step(*args, **kws)
        return_states, return_scores = self.enumerate_graph(ps, self.max_states)
        return_values = [rs.value for rs in return_states]
        return_probs = np.exp(return_scores)
        return_probs = return_probs / return_probs.sum()
        normalized_dist = {}
        for rv, rp in zip(return_values, return_probs):
            normalized_dist[rv] = normalized_dist.get(rv, 0.) + rp
        return Categorical.from_dict(normalized_dist)


    def enumerate_graph(
        self,
        init_ps: ProgramState,
        max_states: int = float('inf'),
    ) -> Dict[ReturnValue, float]:
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

            # we enumerate successors of a program state differently depending on
            # what kind of state it is
            if isinstance(ps, SampleState):
                successors, scores = self.enumerate_sample_state_successors(ps)
            elif isinstance(ps, EnterCallState):
                successors, scores = self.enumerate_enter_call_state_successors(ps)
            elif isinstance(ps, InitialState):
                new_ps, step_score = self.next_state_score(ps)
                successors = [new_ps]
                scores = [step_score]
            else:
                raise ValueError("Unrecognized program state message")

            # logic for updating graph
            for new_ps, score in zip(successors, scores):
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
        )[0]
        return return_states, np.log(sp_return_probs.toarray().flatten())


    def next_state_score(
        self,
        init_ps: ProgramState,
        value=None
    ) -> Tuple[ProgramState, float]:
        if isinstance(init_ps, SampleState):
            ps = init_ps.step(value)
        else:
            ps = init_ps.step()
        score = 0.
        while not isinstance(ps, (SampleState, EnterCallState, ReturnState, ExitCallState)):
            if isinstance(ps, ObserveState):
                score += ps.distribution.log_probability(ps.value)
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
            new_ps, new_score = self.next_state_score(ps, value)
            score += new_score
            successors.append(new_ps)
            scores.append(score)
        return successors, scores


    def enumerate_enter_call_state_successors(
        self,
        init_ps: EnterCallState,
    ) -> Tuple[List[ProgramState], List[float]]:
        # when we enter a call, we take the first step, see if it returns immediately
        # and then take the subsequent step
        # if not, we enumerate the graph to get all the exit states, then for
        # each exit state, we take the next deterministic step
        ps, init_score = self.next_state_score(init_ps)
        if isinstance(ps, ExitCallState):
            return [ps.step()], [init_score]
        successors, scores = self.enumerate_graph(init_ps=ps, max_states=self.max_states)
        successors = [rs.step() for rs in successors]
        scores = [init_score + score for score in scores]
        return successors, scores

    def _run_with_stats(self, *args, **kws) -> Tuple[Categorical, EnumerationStats]:
        self._stats = EnumerationStats()
        result = self.run(*args, **kws)
        self._stats, stats = None, self._stats
        return result, stats
