import math
import dataclasses
from collections import defaultdict
from typing import Mapping, Hashable, Callable, Any, List, Union, Dict

from gorgo.core import ReturnState, SampleState, ObserveState, ProgramState
from gorgo.interpreter import CPSInterpreter
from gorgo.distributions import Categorical, RandomNumberGenerator, Distribution, \
    Dirichlet
from gorgo.distributions.random import default_rng
from gorgo.distributions.support import Simplex

@dataclasses.dataclass
class Entry:
    name : Hashable
    distribution : Distribution
    value : Any
    log_prob : float
    is_sample : bool
    order : int = None
    is_multivariate : bool = False
    program_state : ProgramState = None

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
            is_sample=isinstance(program_state, SampleState),
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
            is_sample=False,
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

class MetropolisHastings:
    """
    Single site Metropolis-Hastings as described by van de Meent et al. (2021)
    Algorithms 6 and 14.
    """
    def __init__(
        self,
        function : Callable,
        samples : int,
        burn_in : int = 0,
        thinning : int = 1,
        seed : int = None,
        uniform_drift_kernel_width : float = None,
        save_diagnostics = False
    ):
        self.function = function
        self.samples = samples
        self.seed= seed
        self.burn_in = burn_in
        self.thinning = thinning
        self.uniform_drift_kernel_width = uniform_drift_kernel_width
        self.save_diagnostics = save_diagnostics

    def run(self, *args, **kws):
        dist, _ = self._run(*args, **kws)
        return dist

    def _run(self, *args, **kws):
        # van de Meent et al. (2018), Algorithm 14
        rng = RandomNumberGenerator(self.seed)
        return_counts = defaultdict(int)
        init_ps = CPSInterpreter().initial_program_state(self.function)
        init_ps = init_ps.step(*args, **kws)

        trace = self.sample_initial_trace(init_ps, rng)
        def sample_site_callback(ps : SampleState):
            if ps.name in trace and ps.name != name:
                return trace[ps.name].value
            else:
                return self.proposal(
                    program_state=ps,
                    cur_value=trace[ps.name].value if ps.name in trace else None,
                    rng=rng
                )

        diagnostics = MHDiagnostics()

        for i in range(self.burn_in + self.samples*self.thinning):
            name = rng.sample([e.name for e in trace.values() if e.is_sample], k=1)[0]
            new_trace = Trace.run_from(
                ps=trace[name].program_state,
                old_trace=trace,
                sample_site_callback=sample_site_callback,
            )

            if new_trace.total_score == float('-inf'):
                log_acceptance_ratio = float('-inf')
            else:
                log_acceptance_ratio = self.calc_log_acceptance_ratio(name, new_trace, trace)
            accept = math.log(rng.random()) < log_acceptance_ratio
            if self.save_diagnostics:
                diagnostics.append(dict(
                    log_acceptance_ratio=log_acceptance_ratio,
                    accept=accept,
                    name=name,
                    new_trace=new_trace,
                    old_trace=trace,
                    burn_in = i < self.burn_in,
                    sampled_trace = i >= self.burn_in and i % self.thinning != 0,
                ))
            if accept:
                trace = new_trace
            if i >= self.burn_in and i % self.thinning == 0:
                return_counts[trace.return_value] += 1
        assert sum(return_counts.values()) == self.samples, (sum(return_counts.values()), self.samples)
        return (
            Categorical.from_dict({e: c/self.samples for e, c in return_counts.items()}),
            diagnostics
        )

    def sample_initial_trace(
        self,
        init_ps : SampleState,
        rng : RandomNumberGenerator
    ) -> Mapping[Hashable, Entry]:
        while True:
            trace = Trace.run_from(
                ps=init_ps,
                sample_site_callback=lambda ps : ps.distribution.sample(rng=rng),
            )
            if trace.total_score > float('-inf'):
                return trace

    def proposal(
        self,
        program_state : SampleState,
        cur_value : Any,
        rng : RandomNumberGenerator
    ):
        if self.uniform_drift_kernel_width is None or cur_value is None:
            return program_state.distribution.sample(rng=rng)
        support = program_state.distribution.support
        if isinstance(support, Simplex):
            u = Dirichlet([1, 1, 1]).sample(rng=rng)
            u = [(ui - 1/len(u))*self.uniform_drift_kernel_width for ui in u]
            return tuple(vi + ui for vi, ui in zip(cur_value, u))
        return program_state.distribution.sample(rng=rng)

    def calc_log_acceptance_ratio(
        self,
        sample_name : Hashable,
        new_db : Trace,
        db : Trace
    ):
        # van de Meent et al. (2018), Algorithm 6

        # We first identify sample states. It's important to ensure we only
        # filter out sample states, since observations must always be
        # included. This is more explicit in Equation 4.21 than Algorithm 6.
        new_db_sample_states = {k for k, e in new_db.items() if e.is_sample}
        db_sample_states = {k for k, e in db.items() if e.is_sample}

        # We filter sample states to those sampled by the proposal, which
        # are the entries unique to each DB.
        new_db_sampled = {sample_name} | (new_db_sample_states - db_sample_states)
        db_sampled = {sample_name} | (db_sample_states - new_db_sample_states)

        # The proposal starts by randomly sampling a name. This is the ratio for the proposals.
        log_acceptance_ratio = math.log(len(db_sample_states)) - math.log(len(new_db_sample_states))

        # For every entry, we incorporate the log probability from samples and observations, filtering
        # out those that were sampled in the proposal, since the term from the log probability and
        # proposal would cancel out.
        for entry in new_db.values():
            if entry.name in new_db_sampled:
                continue
            log_acceptance_ratio += entry.log_prob
        for entry in db.values():
            if entry.name in db_sampled:
                continue
            log_acceptance_ratio -= entry.log_prob
        return log_acceptance_ratio

@dataclasses.dataclass
class MHDiagnostics:
    history : List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def append(self, **kws):
        self.history.append(kws)

    @property
    def sampled_traces(self):
        return [e for e in self.history if e['sampled_trace']]

    @property
    def acceptance_ratio(self):
        accepted = [e['accept'] for e in self.sampled_traces]
        return sum(accepted) / len(accepted)

    @property
    def sampled_traces(self):
        return [e['new_trace'] if e['accept'] else e['old_trace'] for e in self.sampled_traces]
