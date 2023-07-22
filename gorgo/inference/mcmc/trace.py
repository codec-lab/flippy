import dataclasses
from typing import Mapping, Hashable, Callable, Any, List, Union, Dict, Tuple

from gorgo.core import ReturnState, SampleState, ObserveState, ProgramState, VariableName, SampleValue
from gorgo.distributions import Categorical, RandomNumberGenerator, Distribution, \
    Dirichlet
from gorgo.distributions.random import default_rng

@dataclasses.dataclass
class Entry:
    name : VariableName
    distribution : Distribution
    value : SampleValue
    log_prob : float
    order : int = None
    is_multivariate : bool = False
    program_state : ProgramState = None

    @property
    def is_observed(self) -> bool:
        return isinstance(self.program_state, ObserveState)

    @property
    def is_return(self) -> bool:
        return isinstance(self.program_state, ReturnState)

    @property
    def is_sample(self) -> bool:
        return isinstance(self.program_state, SampleState)

class Trace:
    def __init__(self):
        self._entries : List[Entry] = []
        self._entry_name_order : Mapping[Hashable, int] = {}

    @staticmethod
    def default_sample_site_value(ps : SampleState) -> Any:
        return ps.distribution.sample(default_rng)

    @staticmethod
    def default_observe_site_value(ps : ObserveState,) -> Any:
        return ps.value

    @staticmethod
    def run_from(
        ps : ProgramState,
        old_trace : 'Trace' = None,
        sample_site_callback : Callable[[SampleState], Any] = None,
        observe_site_callback : Callable[[ObserveState], Any] = None,
        break_early : bool = False
    ) -> 'Trace':
        new_trace = Trace()
        if old_trace is not None and len(old_trace) > 0:
            if ps.name not in old_trace:
                raise ValueError(f"Name {ps.name} not already in trace")
            new_trace._entries = old_trace._entries[:old_trace._entry_name_order[ps.name]]
            new_trace._entry_name_order = {
                e.name : i for i, e in enumerate(new_trace._entries)
            }

        if sample_site_callback is None:
            sample_site_callback = Trace.default_sample_site_value
        if observe_site_callback is None:
            observe_site_callback = Trace.default_observe_site_value
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
            if break_early and new_trace._entries[-1].log_prob == float('-inf'):
                break
        return new_trace

    def add_site(
        self,
        program_state : Union[SampleState, ObserveState],
        value : Any,
    ):
        name = program_state.name
        log_prob = program_state.distribution.log_probability(value)
        is_multivariate = False
        # (
        #     hasattr(program_state, "distribution") and \
        #     isinstance(program_state.distribution, Multivariate) and \
        #     hasattr(value, "__len__")
        # )
        self._entries.append(Entry(
            name=name,
            order=len(self),
            distribution=program_state.distribution,
            value=value,
            log_prob=log_prob,
            is_multivariate=is_multivariate,
            program_state=program_state
        ))
        self._entry_name_order[name] = len(self._entries) - 1

    def add_return_state(self, program_state : ReturnState):
        self._entries.append(Entry(
            name=program_state.name,
            order=len(self),
            distribution=None,
            value=program_state.value,
            log_prob=0.0,
            is_multivariate=False,
            program_state=program_state
        ))
        self._entry_name_order[program_state.name] = len(self._entries) - 1

    @property
    def return_value(self):
        assert isinstance(self._entries[-1].program_state, ReturnState), \
            ("Trace does not have a return value", self._entries)
        return self._entries[-1].program_state.value

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key):
        return key in self._entry_name_order

    @property
    def total_score(self):
        if self._entries[-1].log_prob == float('-inf'):
            return float('-inf')
        return sum(e.log_prob for e in self._entries)

    def entries(self, start_name : Hashable = None):
        if start_name is None:
            start_idx = 0
        else:
            start_idx = self._entry_name_order[start_name]
        for e in self._entries[start_idx:]:
            yield e

    @property
    def sample_site_names(self) -> List[Hashable]:
        return [e.name for e in self._entries if e.is_sample]

    @property
    def sample_site_entries(self) -> List[Entry]:
        return [e for e in self._entries if e.is_sample]

    def items(self):
        for name, order in self._entry_name_order.items():
            yield name, self._entries[order]

    def values(self):
        for order in self._entry_name_order.values():
            yield self._entries[order]

    def keys(self):
        for name in self._entry_name_order.keys():
            yield name

    def __getitem__(self, key):
        return self._entries[self._entry_name_order[key]]
