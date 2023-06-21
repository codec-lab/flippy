import random

MAX_INT64 = 2**64 - 1

class BaseRandomNumberGenerator(random.Random):
    def fork(self, n : int):
        return [self.__class__(self.randint(0, MAX_INT64)) for _ in range(n)]

    def new_seed(self):
        return self.randint(0, MAX_INT64)

try:
    import numpy as np
    class RandomNumberGenerator(BaseRandomNumberGenerator):
        def __init__(self, seed=None):
            super().__init__(seed)
            self.np = np.random.Generator(np.random.PCG64(self.randint(0, MAX_INT64)))
except ImportError:
    RandomNumberGenerator = BaseRandomNumberGenerator

default_rng = RandomNumberGenerator(None)
