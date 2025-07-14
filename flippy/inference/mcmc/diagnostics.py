import dataclasses
from functools import cached_property
from typing import Mapping, Hashable, Callable, Any, List, Union, Dict, Tuple

from flippy.inference.mcmc.trace import Trace

@dataclasses.dataclass
class MCMCDiagnosticsEntry:
    old_trace : Trace
    new_trace : Trace
    log_acceptance_threshold : float
    log_acceptance_ratio : float
    sampled_trace : Trace
    accept : bool
    save_sample : bool
    auxiliary_vars : Any

@dataclasses.dataclass
class MCMCDiagnostics:
    history : List[MCMCDiagnosticsEntry] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.shortened_names = {}

    def calc_variable_values(self, only_sampled : bool = True):
        variable_values = {}

        if only_sampled:
            history = self.sampled_history
        else:
            history = self.history
        for i, s in enumerate(history):
            if not s.save_sample:
                continue
            for e in s.sampled_trace.entries():
                if not e.is_sample:
                    continue
                if not isinstance(e.name, str) and e.name[0] == (None, None): #HACK
                    name = self.shorten_name(e.name)
                else:
                    name = e.name
                if name not in variable_values:
                    variable_values[name] = [None]*len(history)
                variable_values[name][i] = e.value
        return variable_values

    def append(self, entry : MCMCDiagnosticsEntry):
        self.history.append(entry)

    @cached_property
    def sampled_history(self) -> List[MCMCDiagnosticsEntry]:
        return [e for e in self.history if e.save_sample]

    def shorten_name(self, name):
        if name in self.shortened_names:
            return self.shortened_names[name]
        top_stack_frame = name[-1]
        short_name = ShortVariableName(
            len(name),
            get_function_name(top_stack_frame[0]),
            top_stack_frame[1]
        )
        if short_name in self.shortened_names.values():
            short_name = short_name.increment_version()
            while short_name in self.shortened_names.values():
                short_name = short_name.increment_version()
        self.shortened_names[name] = short_name
        return short_name

    def plot_parameter_samples(
        self,
        top_n_vars = None,
        subplot_height = 8,
        subplot_width = 2,
        sort_varnames = False,
        only_sampled = True,
    ):
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        if top_n_vars in (None, -1):
            top_n_vars = len(self.calc_variable_values(only_sampled=only_sampled))

        df = pd.DataFrame(self.calc_variable_values(only_sampled=only_sampled))
        isna = df.isna().sum(axis=0)
        if sort_varnames:
            varnames = sorted(isna.index, key=lambda x: isna[x])
        else:
            varnames = isna.index
        fig, axes = plt.subplots(top_n_vars, 1, figsize=(subplot_height, subplot_width*top_n_vars))
        for i in range(top_n_vars):
            var_vals = df[varnames[i]].values
            ax = axes[i]
            if all(isinstance(v, (tuple, list)) for v in var_vals if v is not None):
                var_vals = [list(v) if v is not None else None for v in var_vals]
                max_row = max(len(v) for v in var_vals if v is not None)
                var_vals = [v if v is not None else [None]*max_row for v in var_vals]
                var_vals = np.array(var_vals)
            elif all(isinstance(v, str) for v in var_vals if v is not None):
                all_values = sorted(set([v for v in var_vals if v is not None]))
                one_hot = np.zeros((len(var_vals), len(all_values)))
                for j, v in enumerate(var_vals):
                    if v is not None:
                        one_hot[j, all_values.index(v)] = 1
                    else:
                        one_hot[j, :] = np.nan
                var_vals = one_hot
            else:
                var_vals = np.array(var_vals)
            ax.plot(var_vals, 'x-')
            ax.set_title(str(varnames[i]))
            ax.set_xlim(0, len(var_vals))
        plt.tight_layout()
        return fig

    def plot_sampler_stats(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        trace_scores = []
        for e in self.history:
            trace_scores.append(dict(
                accept = e.accept,
                log_acceptance_threshold = e.log_acceptance_threshold,
                log_acceptance_ratio = e.log_acceptance_ratio,
                old_trace = e.old_trace.total_score,
                new_trace = e.new_trace.total_score,
                sampled_trace = e.sampled_trace.total_score,
            ))
        trace_scores = pd.DataFrame(trace_scores)

        fig, axes = plt.subplots(2, 1, figsize=(8, 5))
        trace_scores['accept'].expanding().mean().plot(ax=axes[0])
        axes[0].set_title(f'Cumulative acceptance rate (final: {trace_scores["accept"].mean():.2f})')
        axes[0].set_ylim(0, 1)
        axes[0].set_xlim(-10, len(trace_scores))
        trace_scores['sampled_trace'].plot(ax=axes[1])
        axes[1].plot(
            [r['sampled_trace'] if r['accept'] else None for _, r in trace_scores.iterrows()],
            'xk', markersize=2
        )
        axes[1].set_title('Accepted trace scores')
        axes[1].set_xlim(-10, len(trace_scores))
        plt.tight_layout()

def get_function_name(fn_src):
    lines = fn_src.split('\n')
    for line in lines:
        if 'def ' in line:
            return line.split('def ')[1].split('(')[0]

@dataclasses.dataclass(frozen=True)
class ShortVariableName:
    depth : int
    function_name : str
    lineno : int
    version : int = 0

    def increment_version(self):
        return dataclasses.replace(self, version=self.version+1)

    def __str__(self):
        return f"{self.function_name}.{self.depth}.{self.lineno}.{self.version}"
