import math
import abc
from collections import defaultdict
from typing import Mapping, Hashable, Callable, Any, List, Union, Dict, Tuple

from gorgo.distributions import Categorical, RandomNumberGenerator, Distribution, \
    Dirichlet
from gorgo.distributions.random import default_rng
from gorgo.core import ReturnState, SampleState, ObserveState, ProgramState
from gorgo.interpreter import CPSInterpreter

from gorgo.inference.mcmc.trace import Trace, Entry
from gorgo.inference.mcmc.diagnostics import MCMCDiagnostics, MCMCDiagnosticsEntry

class MarkovChainMonteCarloABC(abc.ABC):
    max_initial_trace_attempts = 10000
    def __init__(
        self,
        function : Callable,
        samples,
        burn_in = 0,
        thinning = 1,
        save_diagnostics = False,
        seed = None
    ):
        self.function = function
        self.samples = samples
        self.burn_in = burn_in
        self.thinning = thinning
        self.save_diagnostics = save_diagnostics
        self.seed = seed

    @abc.abstractmethod
    def trace_proposal(
        self, *,
        old_trace : Trace,
        new_trace : Trace = None,
        rng : RandomNumberGenerator,
        aux : Any = None,
    ) -> Tuple[Trace, float]:
        pass

    @abc.abstractmethod
    def aux_proposal(
        self,
        trace : Trace,
        rng : RandomNumberGenerator,
        aux : Any = None,
    ) -> Tuple[Any, float]:
        pass

    def generate_initial_trace(
        self,
        initial_program_state : ProgramState,
        rng : RandomNumberGenerator
    ) -> Trace:
        for i in range(self.max_initial_trace_attempts):
            trace = Trace.run_from(
                ps=initial_program_state,
                sample_site_callback=lambda ps : ps.distribution.sample(rng=rng),
                observe_site_callback=lambda ps : ps.value
            )
            if trace.total_score > float('-inf'):
                return trace
        raise RuntimeError(f'Could not generate initial trace after {self.max_initial_trace_attempts} attempts')

    def run(self, *args, **kws):
        dist, _ = self._run(*args, **kws)
        return dist

    def _run(self, *args, **kws):
        init_ps = CPSInterpreter().initial_program_state(self.function)
        init_ps = init_ps.step(*args, **kws)
        rng = RandomNumberGenerator(self.seed)
        initial_trace = self.generate_initial_trace(init_ps, rng)
        assert initial_trace.total_score > float('-inf')
        return self.run_from_initial_trace(initial_trace, rng=rng)

    def run_from_initial_trace(self, initial_trace : Trace, rng : RandomNumberGenerator = default_rng):
        diagnostics = MCMCDiagnostics()
        return_counts = defaultdict(int)
        old_trace = initial_trace
        for i in range(self.burn_in + self.samples):
            log_acceptance_threshold = math.log(rng.random())
            aux, log_old_aux_proposal_prob = self.aux_proposal(
                old_trace,
                rng=rng
            )
            new_trace, log_new_trace_proposal_prob = self.trace_proposal(
                old_trace=old_trace,
                rng=rng,
                aux=aux
            )
            _, log_new_aux_proposal_prob = self.aux_proposal(
                new_trace,
                rng=rng,
                aux=aux
            )
            _, log_old_trace_proposal_prob = self.trace_proposal(
                old_trace=new_trace,
                new_trace=old_trace,
                rng=rng,
                aux=aux
            )
            old_score = old_trace.total_score
            new_score = new_trace.total_score
            log_acceptance_ratio = (
                new_score + log_new_aux_proposal_prob + log_old_trace_proposal_prob
            ) - (
                old_score + log_old_aux_proposal_prob + log_new_trace_proposal_prob
            )
            assert not math.isnan(log_acceptance_ratio)
            if log_acceptance_ratio > log_acceptance_threshold:
                old_trace = new_trace
            save_sample = i >= self.burn_in and (i - self.burn_in) % self.thinning == 0
            if save_sample:
                return_counts[old_trace.return_value] += 1
            if self.save_diagnostics:
                diagnostics.append(MCMCDiagnosticsEntry(
                    old_trace=old_trace,
                    new_trace=new_trace,
                    log_acceptance_threshold=log_acceptance_threshold,
                    save_sample=save_sample,
                    auxiliary_vars=aux,
                ))
        assert sum(return_counts.values()) == self.samples, (sum(return_counts.values()), self.samples)
        return (
            Categorical.from_dict({e: c/self.samples for e, c in return_counts.items()}),
            diagnostics
        )
