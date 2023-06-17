import random

class BaseRandomNumberGenerator(random.Random):
    def fork(self, n : int):
        return [self.__class__(self.randint(0, int(1e100))) for _ in range(n)]

    def new_seed(self):
        return self.randint(0, int(1e100))

try:
    import numpy as np
    class RandomNumberGenerator(BaseRandomNumberGenerator):
        def __init__(self, x=None):
            super().__init__(x)
            self.np = np.random.Generator(np.random.PCG64(self.randint(0, 2**32)))
except ImportError:
    RandomNumberGenerator = BaseRandomNumberGenerator

default_rng = RandomNumberGenerator(None)
