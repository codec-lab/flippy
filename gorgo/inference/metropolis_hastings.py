import random
import math
from collections import defaultdict
from typing import Mapping, Hashable, Callable

from gorgo.core import ReturnState, SampleState, ObserveState
from gorgo.interpreter import CPSInterpreter

from collections import namedtuple
Entry = namedtuple("Entry", "name distribution value log_prob is_sample")

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
        seed : int = None
    ):
        self.function = function
        self.samples = samples
        self.seed= seed
        self.burn_in = burn_in
        self.thinning = thinning

    def run(self, *args, **kws):
        # van de Meent et al. (2018), Algorithm 14
        rng = random.Random(self.seed)
        return_counts = defaultdict(int)
        init_ps = CPSInterpreter().initial_program_state(self.function)
        init_ps = init_ps.step(*args, **kws)
        db : Mapping[Hashable, Entry] = {}
        new_db : Mapping[Hashable, Entry] = {}
        for i in range(-1, self.burn_in + self.samples*self.thinning):
            initial_trace = i == -1
            if not initial_trace:
                name = rng.sample([e.name for e in db.values() if e.is_sample], k=1)[0]
            ps = init_ps
            while not isinstance(ps, ReturnState):
                assert ps.name not in new_db, f"Name already in trace: {ps.name}"
                if isinstance(ps, SampleState):
                    if ps.name in db and ps.name != name:
                        value = db[ps.name].value
                    else:
                        value = self.proposal(ps, rng)
                    log_prob = ps.distribution.log_probability(value)
                    new_db[ps.name] = Entry(
                        ps.name, ps.distribution, value, log_prob, True
                    )
                    ps = ps.step(value)
                elif isinstance(ps, ObserveState):
                    log_prob = ps.distribution.log_probability(ps.value)
                    new_db[ps.name] = Entry(
                        ps.name, ps.distribution, ps.value, log_prob, False
                    )
                    ps = ps.step()
            if initial_trace:
                accept = True
            else:
                log_acceptance_ratio = self.calc_log_acceptance_ratio(name, new_db, db)
                accept = math.log(rng.random()) < log_acceptance_ratio
            if accept:
                db, return_val = new_db, ps.value
            new_db = {}
            if i >= self.burn_in and i % self.thinning == 0:
                return_counts[return_val] += 1
        assert sum(return_counts.values()) == self.samples, (sum(return_counts.values()), self.samples)
        return {e: c/self.samples for e, c in return_counts.items()}

    def proposal(
        self,
        program_state : SampleState,
        rng : random.Random
    ):
        return program_state.distribution.sample(rng=rng)

    def calc_log_acceptance_ratio(
        self,
        sample_name : Hashable,
        new_db : Mapping[Hashable, Entry],
        db : Mapping[Hashable, Entry]
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
