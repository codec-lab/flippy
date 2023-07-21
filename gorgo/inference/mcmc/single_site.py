from gorgo.inference.mcmc.base_mcmc import MarkovChainMonteCarloABC
from gorgo.distributions import Bernoulli, Distribution, Categorical, Dirichlet, Uniform
from typing import Mapping, Hashable, Callable, Any, List, Union, Dict, Tuple, Sequence

from gorgo.distributions import RandomNumberGenerator
from gorgo.distributions.base import FiniteDistribution
from gorgo.core import ReturnState, SampleState, ObserveState, ProgramState
from gorgo.distributions.support import ClosedInterval, Simplex

from gorgo.inference.mcmc.trace import Trace, Entry

class SingleSiteMCMC(MarkovChainMonteCarloABC):
    def __init__(
        self,
        function : Callable,
        samples,
        burn_in = 0,
        thinning = 1,
        save_diagnostics = False,
        seed = None,
        use_drift_kernels = False,
        uniform_drift_kernel_width = .5,
        simplex_proposal_kernel_alpha = 10,
    ):
        super().__init__(
            function=function,
            samples=samples,
            burn_in=burn_in,
            thinning=thinning,
            save_diagnostics=save_diagnostics,
            seed=seed
        )
        self.use_drift_kernels = use_drift_kernels
        self.uniform_drift_kernel_width = uniform_drift_kernel_width
        self.simplex_proposal_kernel_alpha = simplex_proposal_kernel_alpha

    def site_proposal_dist(
        self,
        old_value : Any,
        program_state : SampleState,
    ) -> Distribution:
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

    def trace_proposal(
        self, *,
        old_trace : Trace,
        new_trace : Trace = None,
        rng : RandomNumberGenerator,
        aux : Any = None,
    ) -> Tuple[Trace, float]:
        resampling_site_name = aux
        if new_trace is None:
            resampling_ps = old_trace[resampling_site_name].program_state
            total_proposal_log_prob = 0
            def sample_site_callback(ps : SampleState):
                nonlocal total_proposal_log_prob
                if ps.name == resampling_site_name:
                    proposal_dist = self.site_proposal_dist(
                        old_value=old_trace[ps.name].value,
                        program_state=ps
                    )
                    value = proposal_dist.sample(rng=rng)
                    proposal_log_prob = proposal_dist.log_probability(value)
                elif ps.name in old_trace:
                    value = old_trace[ps.name].value
                    proposal_log_prob = 0
                else:
                    value = ps.distribution.sample(rng=rng)
                    proposal_log_prob = ps.distribution.log_probability(value)
                total_proposal_log_prob += proposal_log_prob
                return value
            new_trace = Trace.run_from(
                ps=resampling_ps,
                old_trace=old_trace,
                sample_site_callback=sample_site_callback,
                observe_site_callback=lambda ps : ps.value,
                break_early=False
            )
        else:
            total_proposal_log_prob = 0
            for entry in new_trace.sample_site_entries:
                if entry.name == resampling_site_name:
                    proposal_dist = self.site_proposal_dist(
                        old_value=old_trace[entry.name].value,
                        program_state=entry.program_state
                    )
                    proposal_log_prob = proposal_dist.log_probability(entry.value)
                elif entry.name in old_trace:
                    proposal_log_prob = 0
                else:
                    proposal_log_prob = entry.distribution.log_probability(entry.value)
                total_proposal_log_prob += proposal_log_prob
        return new_trace, total_proposal_log_prob

    def aux_proposal(
        self,
        trace : Trace,
        rng : RandomNumberGenerator,
        aux : Any = None,
    ) -> Tuple[Any, float]:
        sample_site_names = trace.sample_site_names
        if aux is None:
            resampling_site_name = rng.choice(sample_site_names)
        else:
            resampling_site_name = aux
        return resampling_site_name, 1/len(sample_site_names)
