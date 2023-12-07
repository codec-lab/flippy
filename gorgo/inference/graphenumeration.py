import queue
from dataclasses import dataclass
from itertools import product
from collections import defaultdict
from typing import Any, Union, List, Tuple, Dict, Set

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye as sp_eye, coo_array

from gorgo.core import ProgramState, ReturnState, SampleState, ObserveState, InitialState
from gorgo.interpreter import CPSInterpreter
from gorgo.distributions import Categorical
from gorgo.tools import logsumexp
from gorgo.types import ReturnValue
from gorgo.callentryexit import EnterCallState, ExitCallState
from gorgo.map import MapEnter, MapExit
from gorgo.inference.enumeration import EnumerationStats, ProgramStateRecord

class GraphEnumeration:
    def __init__(
        self,
        function,
        max_states=float('inf'),
    ):
        self.function = function
        self.max_states = max_states
        self._stats = None

    def run(self, *args, **kws):
        init_ps = CPSInterpreter().initial_program_state(self.function)
        ps = init_ps.step(*args, **kws)
        return_states, return_scores = self.enumerate_graph(ps, self.max_states)
        if len(return_states) == 0:
            raise ValueError("No return states encountered during enumeration")
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
                successors, scores = self.enumerate_enter_map_state_successors(ps)
            elif isinstance(ps, InitialState):
                new_ps, step_score = self.next_choice_state(ps)
                if step_score > float('-inf'):
                    successors = [new_ps]
                    scores = [step_score]
                else:
                    successors = []
                    scores = []
            else:
                raise ValueError("Unrecognized program state message")

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


    def next_choice_state(
        self,
        init_ps: ProgramState,
        value=None
    ) -> Tuple[ProgramState, float]:
        """
        This runs a program state until it reaches a choice state or an exit
        state. It returns the next choice state and score collected along the way.
        """
        if isinstance(init_ps, (SampleState, MapExit)):
            ps = init_ps.step(value)
        else:
            ps = init_ps.step()
        score = 0.
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
        init_ps: EnterCallState
    ) -> Tuple[List[ExitCallState], List[float]]:
        # when we enter a call, we take the first step, see if it returns immediately
        # and then take the subsequent step
        # if not, we enumerate the graph to get all the exit states, then for
        # each exit state, we take the next deterministic step
        ps, init_score = self.next_choice_state(init_ps)
        if isinstance(ps, ExitCallState):
            return [ps], [init_score]
        successors, scores = self.enumerate_graph(init_ps=ps, max_states=self.max_states)
        scores = [init_score + score for score in scores]
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

    def enumerate_exit_map_state_successors(self, enumerator: GraphEnumeration):
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

    def iteratively_expand(self, enumerator: GraphEnumeration):
        frontier: List[MapCrossProductNode] = [self]
        while frontier:
            node = frontier.pop()
            if node.iteration_terminated():
                continue
            node.expand(enumerator)
            frontier.extend(node.children)

    def expand(self, enumerator: GraphEnumeration):
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
