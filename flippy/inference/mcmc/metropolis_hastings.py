import math
from functools import partial, cached_property
import abc
from collections import defaultdict
from typing import Callable, List, Union, Dict, Tuple, TypeVar, TYPE_CHECKING

from flippy.tools import isclose
from flippy.distributions import Categorical, RandomNumberGenerator, \
    Dirichlet, Uniform, Beta
from flippy.distributions.random import default_rng
from flippy.distributions.base import Distribution, FiniteDistribution
from flippy.distributions.support import ClosedInterval, Simplex
from flippy.core import ReturnState, SampleState, ObserveState, ProgramState
from flippy.interpreter import CPSInterpreter
from flippy.inference.inference import InferenceAlgorithm
from flippy.types import Element

from flippy.inference.mcmc.trace import Trace, Entry
from flippy.inference.mcmc.diagnostics import MCMCDiagnostics, MCMCDiagnosticsEntry

from flippy.types import VariableName, SampleValue

ProposalKernel = Callable[['SampleValue', SampleState], Distribution['SampleValue']]

class MetropolisHastings(InferenceAlgorithm[Element]):
    # this is finite in case it is impossible to initialize a trace
    max_initial_trace_attempts = 1000
    def __init__(
        self,
        function : Callable,
        samples : int,
        burn_in : int = 0,
        thinning : int = 1,
        seed : int = None,
        use_drift_kernels : bool = True,
        uniform_drift_kernel_width : float = 1,
        simplex_proposal_kernel_alpha : float = 10,
        custom_proposal_kernels : Callable[['VariableName'], ProposalKernel] = None,
        custom_initial_trace_kernel : Callable[['VariableName'], 'SampleValue'] = None,
        verbose : bool = False
    ):
        self.function = function
        self.samples = samples
        self.burn_in = burn_in
        self.thinning = thinning
        self.save_diagnostics = False
        self.seed = seed
        self.use_drift_kernels = use_drift_kernels
        self.uniform_drift_kernel_width = uniform_drift_kernel_width
        self.simplex_proposal_kernel_alpha = simplex_proposal_kernel_alpha
        self.custom_proposal_kernels = custom_proposal_kernels
        self.custom_initial_trace_kernel = custom_initial_trace_kernel
        self.verbose = verbose

    def run(self, *args, **kws) -> Distribution[Element]:
        self.save_diagnostics = False
        dist, _ = self._run(*args, **kws)
        return dist

    def run_with_diagnostics(self, *args, **kws) -> Tuple[Distribution, MCMCDiagnostics]:
        self.save_diagnostics = True
        dist, diagnostics = self._run(*args, **kws)
        return dist, diagnostics

    def _run(self, *args, **kws):
        init_ps = self.initial_program_state
        init_ps = init_ps.step(*args, **kws)
        rng = RandomNumberGenerator(self.seed)
        initial_trace = self.generate_initial_trace(init_ps, rng)
        assert initial_trace.total_score > float('-inf')
        return self.run_from_initial_trace(initial_trace, rng=rng)

    @cached_property
    def initial_program_state(self):
        return CPSInterpreter().initial_program_state(self.function)

    def run_from_initial_trace(
        self,
        initial_trace : Trace,
        rng : RandomNumberGenerator = default_rng
    ):
        diagnostics = MCMCDiagnostics()
        return_counts = defaultdict(int)
        old_trace = initial_trace
        iterator = range(self.burn_in + self.samples*self.thinning)
        if self.verbose:
            from tqdm.notebook import tqdm
            print('Running MCMC')
            iterator = tqdm(iterator)
        for i in iterator:
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
            _target_site_name, new_site_score = \
                self.choose_target_site(
                    trace=new_trace,
                    target_site_name=target_site_name,
                    rng=rng
                )
            assert target_site_name == _target_site_name
            _, old_proposal_score = \
                self.choose_new_trace(
                    old_trace=new_trace,
                    target_site_name=target_site_name,
                    new_trace=old_trace,
                    rng=rng
                )
            log_acceptance_ratio = self.calc_log_acceptance_ratio(
                new_score=new_trace.total_score,
                old_score=old_trace.total_score,
                new_site_score=new_site_score,
                old_proposal_score=old_proposal_score,
                old_site_score=old_site_score,
                new_proposal_score=new_proposal_score
            )
            log_acceptance_threshold = math.log(rng.random())
            accept = log_acceptance_ratio > log_acceptance_threshold

            if accept:
                old_trace = new_trace
            save_sample = (i >= self.burn_in) and (((i - self.burn_in) % self.thinning) == 0)
            if save_sample:
                return_counts[old_trace.return_value] += 1
            if self.save_diagnostics:
                diagnostics.append(MCMCDiagnosticsEntry(
                    old_trace=old_trace,
                    new_trace=new_trace,
                    sampled_trace=new_trace if accept else old_trace,
                    accept=accept,
                    log_acceptance_threshold=log_acceptance_threshold,
                    log_acceptance_ratio=log_acceptance_ratio,
                    save_sample=save_sample,
                    auxiliary_vars=target_site_name,
                ))
        assert sum(return_counts.values()) == self.samples, (sum(return_counts.values()), self.samples)
        return (
            Categorical.from_dict({e: c/self.samples for e, c in return_counts.items()}),
            diagnostics
        )

    def calc_log_acceptance_ratio(
        self,
        new_score,
        old_score,
        new_site_score,
        old_proposal_score,
        old_site_score,
        new_proposal_score,
    ):
        log_acceptance_num = new_score + new_site_score + old_proposal_score
        log_acceptance_den = old_score + old_site_score + new_proposal_score
        if log_acceptance_num == float('-inf'):
            return float('-inf')
        else:
            log_acceptance_ratio = log_acceptance_num - log_acceptance_den
            assert not math.isnan(log_acceptance_ratio)
            return log_acceptance_ratio

    def generate_initial_trace(
        self,
        initial_program_state : ProgramState,
        rng : RandomNumberGenerator = default_rng
    ) -> Trace:
        def sample_site_callback(ps : SampleState) -> 'SampleValue':
            assert not ps.fit, f"MetropolisHastings doesn't support Distribution.fit: {ps.name}"
            if ps.initial_value in ps.distribution.support:
                value = ps.initial_value
            elif self.custom_initial_trace_kernel is not None:
                value = self.custom_initial_trace_kernel(ps.name)
                if value is None:
                    value = ps.distribution.sample(rng=rng)
            else:
                value = ps.distribution.sample(rng=rng)
            return value

        iterator = range(self.max_initial_trace_attempts)
        if self.verbose:
            from tqdm.notebook import tqdm
            print('Generating initial trace')
            iterator = tqdm(iterator)
        for i in iterator:
            trace = Trace.run_from(
                ps=initial_program_state,
                old_trace=None,
                sample_site_callback=sample_site_callback,
                observe_site_callback=lambda ps : ps.value
            )
            if trace.total_score > float('-inf'):
                return trace
        raise RuntimeError(f'Could not generate initial trace after {self.max_initial_trace_attempts} attempts')

    def choose_target_site(
        self,
        trace : Trace,
        target_site_name : 'VariableName' = None,
        rng : RandomNumberGenerator = default_rng,
    ) -> Tuple['VariableName', float]:
        sample_sites = [e.name for e in trace.entries() if e.is_sample]
        if target_site_name is None:
            return rng.choice(sample_sites), math.log(1/len(sample_sites))
        else:
            log_prob = math.log(1/len(sample_sites)) if target_site_name in sample_sites else float('-inf')
            return target_site_name, log_prob

    def choose_new_trace(
        self,
        old_trace : Trace,
        target_site_name : 'VariableName',
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
        target_site_name : 'VariableName',
        rng=default_rng
    ) -> Trace:
        def sample_site_callback(ps : SampleState):
            assert not ps.fit, f"MetropolisHastings doesn't support Distribution.fit: {ps.name}"
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
        target_site_name : 'VariableName',
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
            assert not math.isnan(total_proposal_log_prob)
        return total_proposal_log_prob

    def site_proposal_dist(
        self,
        old_value : 'SampleValue',
        program_state : SampleState,
    ) -> Distribution['SampleValue']:
        if self.custom_proposal_kernels is not None:
            proposal_function = self.custom_proposal_kernels(program_state.name)
            if proposal_function is not None:
                return proposal_function(old_value, program_state)

        if not self.use_drift_kernels:
            return program_state.distribution

        if isinstance(program_state.distribution, Beta):
            if isclose(old_value, 0) or isclose(old_value, 1):
                return program_state.distribution
            # This is the Beta version of the Dirichlet kernel below
            return Beta(
                alpha=old_value*self.simplex_proposal_kernel_alpha,
                beta=(1 - old_value)*self.simplex_proposal_kernel_alpha
            )
        elif isinstance(program_state.distribution.support, ClosedInterval):
            # Note that this should be automatically clipped for values out of
            # bounds by setting the score to -inf
            return Uniform(
                old_value - self.uniform_drift_kernel_width/2,
                old_value + self.uniform_drift_kernel_width/2
            )
        elif isinstance(program_state.distribution.support, Simplex):
            # This implementation is borrowed from WebPPL
            return Dirichlet([
                v*self.simplex_proposal_kernel_alpha for v in old_value
            ])
        elif isinstance(program_state.distribution, FiniteDistribution):
            return Categorical(program_state.distribution.support)
        else:
            return program_state.distribution
