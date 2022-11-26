import random
from gorgo.core import StartingMessage, SampleMessage, ObserveMessage
from gorgo.interpreter import CPSInterpreter

class SamplePrior:
    def __init__(self, function):
        self.function = function
    
    def run(self, *args, rng=random, **kws):
        ps = CPSInterpreter().initial_program_state(self.function)
        trajectory = [ps]
        while not ps.is_returned():
            if isinstance(ps.message, StartingMessage):
                ps = ps.step(*args, **kws)
            elif isinstance(ps.message, SampleMessage):
                value = ps.message.distribution(rng=rng)
                ps = ps.step(value)
            elif isinstance(ps.message, ObserveMessage):
                ps = ps.step()
            else:
                raise ValueError("Unrecognized program state message")
            trajectory.append(ps)
        return trajectory