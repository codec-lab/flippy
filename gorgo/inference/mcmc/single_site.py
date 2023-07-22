import math
import abc
from functools import cached_property
from collections import defaultdict
from typing import Mapping, Hashable, Callable, Any, List, Union, Dict, Tuple

from gorgo.distributions import Categorical, RandomNumberGenerator, \
    Dirichlet, Uniform
from gorgo.distributions.random import default_rng
from gorgo.distributions.base import Distribution, FiniteDistribution
from gorgo.distributions.support import ClosedInterval, Simplex
from gorgo.core import ReturnState, SampleState, ObserveState, ProgramState
from gorgo.interpreter import CPSInterpreter

from gorgo.inference.mcmc.trace import Trace, Entry
from gorgo.inference.mcmc.diagnostics import MCMCDiagnostics, MCMCDiagnosticsEntry

class SingleSiteMetropolisHastings:
    max_initial_trace_attempts = 1000
    def __init__(
        self,
        function : Callable,
        samples,
        burn_in = 0,
        thinning = 1,
        save_diagnostics = False,
        seed = None,
        use_drift_kernels = True,
        uniform_drift_kernel_width = 1,
        simplex_proposal_kernel_alpha = 10,
        custom_proposal_kernels = None,
    ):
        self.function = function
        self.samples = samples
        self.burn_in = burn_in
        self.thinning = thinning
        self.save_diagnostics = save_diagnostics
        self.seed = seed
        self.use_drift_kernels = use_drift_kernels
        self.uniform_drift_kernel_width = uniform_drift_kernel_width
        self.simplex_proposal_kernel_alpha = simplex_proposal_kernel_alpha
        self.custom_proposal_kernels = custom_proposal_kernels

    def run(self, *args, **kws) -> Distribution:
        dist, _ = self.run_with_diagnostics(*args, **kws)
        return dist

    def run_with_diagnostics(self, *args, **kws) -> Tuple[Distribution, MCMCDiagnostics]:
        init_ps = CPSInterpreter().initial_program_state(self.function)
        init_ps = init_ps.step(*args, **kws)
        rng = RandomNumberGenerator(self.seed)
        initial_trace = self.generate_initial_trace(init_ps, rng)
        assert initial_trace.total_score > float('-inf')
        return self.run_from_initial_trace(initial_trace, rng=rng)

    def run_from_initial_trace(
        self,
        initial_trace : Trace,
        rng : RandomNumberGenerator = default_rng
    ):
        diagnostics = MCMCDiagnostics()
        return_counts = defaultdict(int)
        old_trace = initial_trace
        for i in range(self.burn_in + self.samples*self.thinning):
            target_site_name, old_site_score = \
                self.choose_target_site(
                    trace=old_trace,
                    rng=rng
                )
            new_trace, new_proposal_score = \
                self.choose_new_trace(
                    old_trace=old_trace,
                    target_site_name=target_site_name,
                    rng=rng
                )
            _, new_site_score = \
                self.choose_target_site(
                    trace=new_trace,
                    target_site_name=target_site_name,
                    rng=rng
                )
            _, old_proposal_score = \
                self.choose_new_trace(
                    old_trace=new_trace,
                    target_site_name=target_site_name,
                    new_trace=old_trace,
                    rng=rng
                )
            new_score = new_trace.total_score
            old_score = old_trace.total_score
            log_acceptance_ratio = (
                new_score + new_site_score + old_proposal_score
            ) - (
                old_score + old_site_score + new_proposal_score
            )
            log_acceptance_threshold = math.log(rng.random())
            assert not math.isnan(log_acceptance_ratio)
            if log_acceptance_ratio > log_acceptance_threshold:
                old_trace = new_trace
            save_sample = (i >= self.burn_in) and (((i - self.burn_in) % self.thinning) == 0)
            if save_sample:
                return_counts[old_trace.return_value] += 1
            if self.save_diagnostics:
                diagnostics.append(MCMCDiagnosticsEntry(
                    old_trace=old_trace,
                    new_trace=new_trace,
                    log_acceptance_threshold=log_acceptance_threshold,
                    save_sample=save_sample,
                    auxiliary_vars=target_site_name,
                ))
        assert sum(return_counts.values()) == self.samples, (sum(return_counts.values()), self.samples)
        return (
            Categorical.from_dict({e: c/self.samples for e, c in return_counts.items()}),
            diagnostics
        )

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

    def choose_target_site(
        self,
        trace : Trace,
        target_site_name : Hashable = None,
        rng : RandomNumberGenerator = default_rng,
    ) -> Tuple[Hashable, float]:
        sample_sites = [e.name for e in trace.values() if e.is_sample]
        if target_site_name is None:
            return rng.choice(sample_sites), math.log(1/len(sample_sites))
        else:
            log_prob = math.log(1/len(sample_sites)) if target_site_name in sample_sites else float('-inf')
            return target_site_name, log_prob

    def choose_new_trace(
        self,
        old_trace : Trace,
        target_site_name : Hashable,
        new_trace : Trace = None,
        rng : RandomNumberGenerator = default_rng,
    ) -> Tuple[Trace, float]:
        if new_trace is None:
            new_trace = self.sample_new_trace(
                old_trace=old_trace,
                target_site_name=target_site_name,
                rng=rng
            )
        log_prob = self.calc_new_trace_log_probability(
            old_trace=old_trace,
            target_site_name=target_site_name,
            new_trace=new_trace
        )
        return new_trace, log_prob

    def sample_new_trace(
        self,
        old_trace : Trace,
        target_site_name : Hashable,
        rng=default_rng
    ) -> Trace:
        def sample_site_callback(ps : SampleState):
            if ps.name == target_site_name:
                proposal_dist = self.site_proposal_dist(
                    old_value=old_trace[ps.name].value,
                    program_state=ps
                )
                value = proposal_dist.sample(rng=rng)
            elif ps.name in old_trace:
                value = old_trace[ps.name].value
            else:
                value = ps.distribution.sample(rng=rng)
            return value

        new_trace = old_trace.run_from(
            ps=old_trace[target_site_name].program_state,
            old_trace=old_trace,
            sample_site_callback=sample_site_callback,
            observe_site_callback=lambda ps : ps.value,
        )
        return new_trace

    def calc_new_trace_log_probability(
        self,
        old_trace : Trace,
        target_site_name : Hashable,
        new_trace : Trace
    ) -> float:
        total_proposal_log_prob = 0
        for entry in new_trace.entries(target_site_name):
            if not entry.is_sample:
                continue
            ps : SampleState = entry.program_state
            if entry.name == target_site_name:
                proposal_dist = self.site_proposal_dist(
                    old_value=old_trace[ps.name].value,
                    program_state=ps
                )
                proposal_log_prob = proposal_dist.log_probability(entry.value)
            elif ps.name in old_trace:
                proposal_log_prob = 0
            else:
                proposal_log_prob = ps.distribution.log_probability(entry.value)
            total_proposal_log_prob += proposal_log_prob
        return total_proposal_log_prob

    def site_proposal_dist(
        self,
        old_value : Any,
        program_state : SampleState,
    ) -> Distribution:
        if self.custom_proposal_kernels is not None:
            proposal_function = self.custom_proposal_kernels(program_state.name)
            if proposal_function is not None:
                return proposal_function(old_value, program_state)

        if not self.use_drift_kernels:
            return program_state.distribution

        support = program_state.distribution.support
        if isinstance(support, ClosedInterval):
            return Uniform(
                old_value - self.uniform_drift_kernel_width/2,
                old_value + self.uniform_drift_kernel_width/2
            )
        elif isinstance(support, Simplex):
            return Dirichlet([
                v*self.simplex_proposal_kernel_alpha for v in old_value
            ])
        elif isinstance(program_state.distribution, FiniteDistribution):
            return Categorical(support)
        else:
            return program_state.distribution
