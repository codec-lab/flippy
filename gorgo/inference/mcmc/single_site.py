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

class SingleSiteMCMC:
    max_initial_trace_attempts = 10000
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
    ):
        self.function = function
        self.samples = samples
        self.burn_in = burn_in
        self.thinning = thinning
        self.save_diagnostics = save_diagnostics
        self.seed = seed
        self.proposal_params = dict(
            use_drift_kernels=use_drift_kernels,
            uniform_drift_kernel_width=uniform_drift_kernel_width,
            simplex_proposal_kernel_alpha=simplex_proposal_kernel_alpha,
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

    def run_from_initial_trace(self, initial_trace : Trace, rng : RandomNumberGenerator = default_rng):
        diagnostics = MCMCDiagnostics()
        return_counts = defaultdict(int)
        old_trace = initial_trace
        for i in range(self.burn_in + self.samples*self.thinning):
            old_site_dist = ResamplingSiteDistribution(old_trace)
            site_name = old_site_dist.sample(rng=rng)
            old_site_score = old_site_dist.log_probability(site_name)
            new_proposal_dist = TraceProposalDistribution(
                old_trace=old_trace,
                resampling_site_name=site_name,
                **self.proposal_params
            )
            new_trace = new_proposal_dist.sample(rng=rng)
            new_proposal_score = new_proposal_dist.log_probability(new_trace)
            new_site_dist = ResamplingSiteDistribution(new_trace)
            new_site_score = new_site_dist.log_probability(site_name)
            old_proposal_dist = TraceProposalDistribution(
                old_trace=new_trace,
                resampling_site_name=site_name,
                **self.proposal_params
            )
            old_proposal_score = old_proposal_dist.log_probability(old_trace)
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
                    auxiliary_vars=site_name,
                ))
        assert sum(return_counts.values()) == self.samples, (sum(return_counts.values()), self.samples)
        return (
            Categorical.from_dict({e: c/self.samples for e, c in return_counts.items()}),
            diagnostics
        )

class TraceProposalDistribution(Distribution):
    def __init__(
        self,
        old_trace : Trace,
        resampling_site_name : Hashable,
        use_drift_kernels : bool,
        simplex_proposal_kernel_alpha : float,
        uniform_drift_kernel_width : float,
        custom_kernels : Callable[[Hashable], Callable[[Any, SampleState], Distribution]] = None,
    ):
        self.old_trace = old_trace
        self.resampling_site_name = resampling_site_name
        self.use_drift_kernels = use_drift_kernels
        self.simplex_proposal_kernel_alpha = simplex_proposal_kernel_alpha
        self.uniform_drift_kernel_width = uniform_drift_kernel_width
        self.custom_kernels = custom_kernels

    def site_proposal_dist(
        self,
        old_value : Any,
        program_state : SampleState,
    ) -> Distribution:
        if self.custom_kernels is not None:
            proposal_function = self.custom_kernels(program_state.name)
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

    def sample(self, rng=default_rng, name=None) -> Trace:
        def sample_site_callback(ps : SampleState):
            if ps.name == self.resampling_site_name:
                proposal_dist = self.site_proposal_dist(
                    old_value=self.old_trace[ps.name].value,
                    program_state=ps
                )
                value = proposal_dist.sample(rng=rng)
            elif ps.name in self.old_trace:
                value = self.old_trace[ps.name].value
            else:
                value = ps.distribution.sample(rng=rng)
            return value

        new_trace = self.old_trace.run_from(
            ps=self.old_trace[self.resampling_site_name].program_state,
            old_trace=self.old_trace,
            sample_site_callback=sample_site_callback,
            observe_site_callback=lambda ps : ps.value,
        )
        return new_trace

    def log_probability(self, new_trace : Trace) -> float:
        total_proposal_log_prob = 0
        for entry in new_trace.entries(self.resampling_site_name):
            if not entry.is_sample:
                continue
            ps : SampleState = entry.program_state
            if entry.name == self.resampling_site_name:
                proposal_dist = self.site_proposal_dist(
                    old_value=self.old_trace[ps.name].value,
                    program_state=ps
                )
                proposal_log_prob = proposal_dist.log_probability(entry.value)
            elif ps.name in self.old_trace:
                proposal_log_prob = 0
            else:
                proposal_log_prob = ps.distribution.log_probability(entry.value)
            total_proposal_log_prob += proposal_log_prob
        return total_proposal_log_prob

    @cached_property
    def support(self):
        return TraceProposalSupport(self)

class TraceProposalSupport:
    def __init__(self, trace_proposal_distribution : "TraceProposalDistribution"):
        self.trace_proposal_distribution = trace_proposal_distribution

    def __contains__(self, element):
        return self.trace_proposal_distribution.log_probability(element) > float('-inf')

class ResamplingSiteDistribution(FiniteDistribution):
    def __init__(self, trace : Trace):
        self.trace = trace

    def sample(self, rng=default_rng, name=None) -> Trace:
        return rng.choice([e.name for e in self.trace.values() if e.is_sample])

    def log_probability(self, element):
        return math.log(1/len([e for e in self.trace.values() if e.is_sample]))
