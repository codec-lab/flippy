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
            else:
                raise ValueError("Unrecognized program state message")
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
        return_scores = self.enumerate_graph(ps, self.max_states)
        return Categorical.from_dict(softmax_dict(return_scores))

    def enumerate_graph(self, ps: ProgramState, max_states: int):
        transition_scores: Dict[Tuple[ProgramState, ProgramState], float] = \
            defaultdict(lambda: float('-inf'))
        frontier: queue.Queue[SampleState] = queue.Queue()
        frontier.put(ps)
        visited : Set[SampleState] = set([ps])
        return_states: List[ReturnState] = []
        state_idx = {ps: 0}
        while not frontier.empty():
            if len(visited) >= max_states:
                break
            ps = frontier.get()
            for value in ps.distribution.support:
                score = ps.distribution.log_probability(value)
                new_ps = ps.step(value)
                while not isinstance(new_ps, (SampleState, ReturnState)):
                    if isinstance(new_ps, ObserveState):
                        score += new_ps.distribution.log_probability(new_ps.value)
                    new_ps = new_ps.step()
                if new_ps not in visited and new_ps not in return_states:
                    if isinstance(new_ps, ReturnState):
                        return_states.append(new_ps)
                    else:
                        state_idx[new_ps] = len(state_idx)
                        visited.add(new_ps)
                    if isinstance(new_ps, SampleState):
                        frontier.put(new_ps)
                transition_scores[(ps, new_ps)] = \
                    logsumexp(transition_scores[(ps, new_ps)], score)
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

        # construct return probability vector
        sp_return_onehot = sp_eye(
            n_states, len(return_states), k=-(n_states - len(return_states))
        ).tocsc()
        eye = sp_eye(n_states).tocsc()
        sp_return_probs = spsolve(
            A=eye - sp_scores_matrix,
            b=sp_return_onehot
        )[0]
        sp_return_probs = sp_return_probs / sp_return_probs.sum()

        # return dictionary of scores
        return_probs = defaultdict(lambda: 0.)
        for ps, prob in zip(return_states, sp_return_probs.toarray().flatten()):
            return_probs[ps.value] += prob
        return_scores = {val: math.log(prob) for val, prob in return_probs.items()}
        return return_scores
