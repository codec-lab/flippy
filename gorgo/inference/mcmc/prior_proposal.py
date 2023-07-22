import math
from gorgo.inference.mcmc.base_mcmc import MarkovChainMonteCarloABC
from typing import Mapping, Hashable, Callable, Any, List, Union, Dict, Tuple

from gorgo.distributions import RandomNumberGenerator
from gorgo.distributions.random import default_rng
from gorgo.core import ReturnState, SampleState, ObserveState, ProgramState

from gorgo.inference.mcmc.trace import Trace, Entry

class PriorProposalMCMC(MarkovChainMonteCarloABC):
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
            def sample_site_callback(ps : SampleState):
                if ps.name == resampling_site_name:
                    value = ps.distribution.sample(rng=rng)
                elif ps.name in old_trace:
                    value = old_trace[ps.name].value
                else:
                    value = ps.distribution.sample(rng=rng)
                return value
            new_trace = Trace.run_from(
                ps=resampling_ps,
                old_trace=old_trace,
                sample_site_callback=sample_site_callback,
                observe_site_callback=lambda ps : ps.value,
                break_early=False
            )
        old_trace_sample_names = set(old_trace.sample_site_names)
        new_trace_sample_names = set(new_trace.sample_site_names)
        resampled_variables = {resampling_site_name} | (new_trace_sample_names - old_trace_sample_names)
        resampled_score = sum(new_trace[n].log_prob for n in resampled_variables)
        return new_trace, resampled_score

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
        return resampling_site_name, math.log(1/len(sample_site_names))
