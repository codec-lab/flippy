import dataclasses
from typing import Mapping, Hashable, Callable, Any, List, Union, Dict, Tuple

from gorgo.inference.mcmc.trace import Trace

@dataclasses.dataclass
class MCMCDiagnosticsEntry:
    old_trace : Trace
    new_trace : Trace
    log_acceptance_threshold : float
    save_sample : bool
    auxiliary_vars : Any

@dataclasses.dataclass
class MCMCDiagnostics:
    history : List[MCMCDiagnosticsEntry] = dataclasses.field(default_factory=list)

    def append(self, entry : MCMCDiagnosticsEntry):
        self.history.append(entry)
