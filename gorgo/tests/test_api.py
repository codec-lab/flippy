from gorgo import infer, Bernoulli, Categorical, recursive_map, recursive_filter, recursive_reduce

def algebra():
    def flip():
        return Bernoulli(0.5).sample()
    def NUM():
        return Categorical(range(5)).sample()
    def OP():
        if flip():
            return '+'
        else:
            return '*'
    def EQ():
        if flip():
            return NUM()
        else:
            return (NUM(), OP(), EQ())
    return EQ()

def test_algebra():
    algebra2 = infer(max_executions=55)(algebra)

    results = algebra2()

    num_ct = 0
    expr_ct = 0
    for e in results.support:
        if isinstance(e, int):
            num_ct += 1
        else:
            assert (
                isinstance(e, tuple) and
                len(e) == 3 and
                isinstance(e[0], int) and
                e[1] in ('*', '+') and
                isinstance(e[2], int)
            )
            expr_ct += 1
    assert num_ct == 5
    assert expr_ct == 5 * 2 * 5

def world_prior():
    return Categorical(range(4)).sample()

def utterance_prior():
    return Categorical([
        'some of the people are nice',
        'all of the people are nice',
        'none of the people are nice',
    ]).sample()

def meaning(utt, world):
    return (
        world > 0 if utt == 'some of the people are nice' else
        world == 3 if utt == 'all of the people are nice' else
        world == 0 if utt == 'none of the people are nice' else
        True
    )

@infer
def literal_listener(utterance):
    world = world_prior()
    m = meaning(utterance, world)
    Bernoulli(1.0).observe(m)
    return world

@infer
def speaker(world) -> str:
    utterance = utterance_prior()
    L = literal_listener(utterance)
    L.observe(world)
    return utterance

@infer
def listener(utterance):
    world = world_prior()
    S = speaker(world)
    S.observe(utterance)
    return world

def test_scalar_implicature():
    assert listener('some of the people are nice').isclose(Categorical(
        [1, 2, 3],
        probabilities=[4/9, 4/9, 1/9],
    ))

def test_builtins():
    @infer
    def model():
        return abs(Categorical([-1, 0, 1]).sample())
    assert model().isclose(Categorical(
        [0, 1],
        probabilities=[1/3, 2/3],
    ))

    @infer
    def model():
        mydict = {}
        return tuple(mydict.items())
    assert model().isclose(Categorical([(),]))

def test_cps_map():
    assert recursive_map(lambda x: x ** 2, []) == []
    assert recursive_map(lambda x: x ** 2, list(range(5))) == [0, 1, 4, 9, 16]

def test_cps_filter():
    assert recursive_filter(lambda x: x % 2 == 0, []) == []
    assert recursive_filter(lambda x: x % 2 == 0, list(range(5))) == [0, 2, 4]

def test_recursive_reduce():
    sumfn = lambda acc, el: acc + el

    assert recursive_reduce(sumfn, [], 0) == 0
    assert recursive_reduce(sumfn, [1], 0) == 1
    assert recursive_reduce(sumfn, [1, 2], 0) == 3
    assert recursive_reduce(sumfn, [1, 2, 3], 0) == 6

    assert recursive_reduce(sumfn, [[3, 4], [5]], []) == [3, 4, 5]
