import dataclasses
from functools import cached_property
from typing import Mapping, Hashable, Callable, List, Union, TYPE_CHECKING

from flippy.core import ReturnState, SampleState, ObserveState, ProgramState
from flippy.distributions import Categorical, RandomNumberGenerator, Distribution, \
    Dirichlet
from flippy.distributions.random import default_rng

from flippy.types import SampleValue, VariableName

@dataclasses.dataclass
class Entry:
    value : 'SampleValue'
    program_state : Union[SampleState, ObserveState] = None

    @property
    def is_sample(self) -> bool:
        return isinstance(self.program_state, SampleState)

    @cached_property
    def score(self) -> float:
        return self.program_state.distribution.log_probability(self.value)

    @property
    def name(self) -> 'VariableName':
        return self.program_state.name

    @property
    def distribution(self) -> Distribution:
        return self.program_state.distribution

class Trace:
    def __init__(self):
        self._entries : List[Entry] = []
        self._entry_name_order : Mapping['VariableName', int] = {}
        self._return_state : ReturnState = None

    @staticmethod
    def run_from(
        ps : ProgramState,
        old_trace : Union['Trace', None],
        sample_site_callback : Callable[[SampleState], 'SampleValue'],
        observe_site_callback : Callable[[ObserveState], 'SampleValue'],
        break_early : bool = True
    ) -> 'Trace':
        new_trace = Trace()
        if old_trace is not None and len(old_trace) > 0:
            if ps.name not in old_trace:
                raise ValueError(f"Name {ps.name} not already in trace")
            new_trace._entries = old_trace._entries[:old_trace._entry_name_order[ps.name]]
            new_trace._entry_name_order = {
                e.name : i for i, e in enumerate(new_trace._entries)
            }

        while True:
            assert ps.name not in new_trace, f"Name {ps.name} already in trace"
            if isinstance(ps, SampleState):
                value = sample_site_callback(ps)
                new_trace.add_site(ps, value)
                ps = ps.step(value)
            elif isinstance(ps, ObserveState):
                new_trace.add_site(ps, observe_site_callback(ps))
                ps = ps.step()
            elif isinstance(ps, ReturnState):
                new_trace.add_return_state(ps)
                break
            if break_early and new_trace._entries[-1].score == float('-inf'):
                break
        return new_trace

    def add_site(
        self,
        program_state : Union[SampleState, ObserveState],
        value : 'SampleValue',
    ):
        self._entries.append(Entry(
            value=value,
            program_state=program_state
        ))
        self._entry_name_order[program_state.name] = len(self._entries) - 1

    def add_return_state(self, program_state : ReturnState):
        self._return_state = program_state

    @property
    def return_value(self):
        assert self._return_state is not None, \
            "Trace does not have a return value"
        return self._return_state.value

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key):
        return key in self._entry_name_order

    @property
    def total_score(self) -> float:
        if self._entries[-1].score == float('-inf'):
            return float('-inf')
        return sum(e.score for e in self._entries)

    def entries(self, start_name : Hashable = None):
        if start_name is None:
            start_idx = 0
        else:
            start_idx = self._entry_name_order[start_name]
        for e in self._entries[start_idx:]:
            yield e

    @property
    def sample_site_names(self) -> List['VariableName']:
        return [e.name for e in self._entries if e.is_sample]

    def __getitem__(self, key):
        return self._entries[self._entry_name_order[key]]
