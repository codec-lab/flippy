import linecache
from joblib import Parallel, delayed

from flippy.inference import Enumeration, LikelihoodWeighting
from flippy.interpreter import CPSInterpreter
from flippy.transforms import CPSFunction
from flippy.distributions.builtin_dists import Uniform, Bernoulli, Categorical
from flippy import flip, draw_from, condition, keep_deterministic, hashabledict
from flippy.tools import isclose

def g(p):
    return flip(p) + flip(p) + flip(p) + flip(p) + flip(p)

def f(p=.5):
    Bernoulli(.9).observe(True) # this will only affect the marginal likelihood
    i = draw_from(range(5))
    j = g(p)
    condition(.1 if i == j else .5)
    return i+j

def h():
    p = Uniform().sample()
    return f(p)

def test_parallel_Enumeration():
    cpu1_dist = Enumeration(f, _cpus=1).run()
    cpu2_dist = Enumeration(f, _cpus=2).run()
    assert cpu1_dist.isclose(cpu2_dist)
    assert isclose(cpu1_dist.marginal_likelihood, cpu2_dist.marginal_likelihood, atol=1e-5)

def test_parallel_LikelihoodWeighting():
    seed = 1391
    cpu1_dist = LikelihoodWeighting(h, samples=5000, _cpus=1, seed=seed).run()
    cpu1_dist = cpu1_dist.marginalize(lambda x: x % 2)
    cpu2_dist = LikelihoodWeighting(h, samples=5000, _cpus=2, seed=seed).run()
    cpu2_dist = cpu2_dist.marginalize(lambda x: x % 2)
    assert cpu1_dist.isclose(cpu2_dist, atol=5e-2), (cpu1_dist, cpu2_dist)
    assert not cpu1_dist.isclose(cpu2_dist, atol=1e-5)

def test_Enumeration__run_partition_enumerates_properly():
    sample_count = [0]
    variable_list = []

    @keep_deterministic
    def inc_sample_count(**variables):
        variable_list.append(hashabledict(variables))
        sample_count[0] += 1

    # special hook for debugging
    inc_sample_count._emit_call_entryexit = False

    def f(p):
        a = Bernoulli(p).sample()
        inc_sample_count(a=a)
        b = Bernoulli(p).sample()
        inc_sample_count(a=a, b=b)
        return a + b

    def run_partitions(f, partitions):
        scs = []
        vls = []
        for idx in range(partitions):
            sample_count[0] = 0
            variable_list[:] = []
            Enumeration(f)._run_partition(.7, _partition_idx=idx, _partitions=partitions, _linecache=linecache.cache)
            scs.append(sample_count[0])
            vls.append(list(variable_list))
        return scs, vls

    scs, vls = run_partitions(f, 2)
    assert scs == [2 + 2] * 2
    prefix = [dict(a=1), dict(a=0)]
    assert vls == [
        prefix + [dict(a=1, b=1), dict(a=1, b=0)],
        prefix + [dict(a=0, b=1), dict(a=0, b=0)],
    ]
    # Checking for completeness, though we need to avoid double-counting the shared prefix
    assert sum(scs) - len(prefix) == len(set([s for partition in vls for s in partition]))

    def f3(p):
        a = Bernoulli(p).sample()
        inc_sample_count(a=a)
        b = Bernoulli(p).sample()
        inc_sample_count(a=a, b=b)
        c = Bernoulli(p).sample()
        inc_sample_count(a=a, b=b, c=c)
        return a + b + c

    # Checking case where frontier exceeds partition count after partitioning
    scs, vls = run_partitions(f3, 2)
    assert scs == [2 + 2 + 4] * 2
    prefix = [dict(a=1), dict(a=0)]
    assert vls == [
        prefix + [dict(a=1, b=1), dict(a=1, b=0)] + [dict(a=1, b=b, c=c) for b in [1, 0] for c in [1, 0]],
        prefix + [dict(a=0, b=1), dict(a=0, b=0)] + [dict(a=0, b=b, c=c) for b in [1, 0] for c in [1, 0]],
    ]
    assert sum(scs) - len(prefix) == len(set([s for partition in vls for s in partition]))

    # Checking case where number of partitions is different than program's branching factor
    scs, vls = run_partitions(f3, 3)
    assert scs == [6, 10, 6]
    prefix = [dict(a=1), dict(a=0), dict(a=1, b=1), dict(a=1, b=0)]
    assert vls == [
        prefix + [dict(a=1, b=1, c=c) for c in [1, 0]],
        prefix + [dict(a=0, b=1), dict(a=0, b=0)] + [dict(a=0, b=b, c=c) for b in [1, 0] for c in [1, 0]],
        prefix + [dict(a=1, b=0, c=c) for c in [1, 0]],
    ]
    assert sum(scs) - 2 * len(prefix) == len(set([s for partition in vls for s in partition]))

def test_CPSFunction_can_be_passed_to_joblib_loky_worker():
    i = 100
    def f():
        return 100 + i
    def m(cps_f2: CPSFunction):
        assert cps_f2()() == 200
    cps_f = CPSInterpreter().non_cps_callable_to_cps_callable(f)
    assert cps_f()() == 200
    Parallel(n_jobs=2, backend="loky")(delayed(m)(cps_f) for _ in range(2))
