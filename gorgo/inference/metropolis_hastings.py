import random
import math
from collections import defaultdict

from gorgo.core import ReturnState, SampleState, ObserveState
from gorgo.interpreter import CPSInterpreter

from collections import namedtuple
Entry = namedtuple("Entry", "name distribution value log_prob is_sample")

class MetropolisHastings:
    def __init__(self, function, samples : int, burn_in=0, thinning=1, seed=None):
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
        db, new_db = {}, {}
        
        for i in range(-1, self.burn_in + self.samples*self.thinning):
            if i > -1:
                name = rng.sample([e.name for e in db.values() if e.is_sample], k=1)[0]
            ps = init_ps
            while not isinstance(ps, ReturnState):
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
            if i == -1:
                accept = True
            else:
                log_acceptance_ratio = self.calc_log_acceptance_ratio(name, new_db, db)
                accept = math.log(rng.random()) < log_acceptance_ratio
            if accept:
                db, return_val = new_db, ps.value
            new_db = {}
            if i > (self.burn_in - 1) and i % self.thinning == 0:
                return_counts[return_val] += 1
        assert sum(return_counts.values()) == self.samples, (sum(return_counts.values()), self.samples)
        return {e: c/self.samples for e, c in return_counts.items()}
    
    def proposal(self, program_state : SampleState, rng : random.Random):
        return program_state.distribution.sample(rng=rng)
    
    def calc_log_acceptance_ratio(self, sample_name, new_db, db):
        # van de Meent et al. (2018), Algorithm 6
        new_db_sampled = {sample_name} | (set(new_db.keys()) - set(db.keys()))
        db_sampled = {sample_name} | (set(db.keys()) - set(new_db.keys()))
        log_acceptance_ratio = math.log(len(db)) - math.log(len(new_db))
        for entry in new_db.values():
            if entry.name in new_db_sampled:
                continue
            log_acceptance_ratio += entry.log_prob
        for entry in db.values():
            if entry.name in db_sampled:
                continue
            log_acceptance_ratio -= entry.log_prob
        return log_acceptance_ratio