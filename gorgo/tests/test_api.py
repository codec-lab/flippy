from gorgo import keep_deterministic, infer, Bernoulli, Multinomial, observe

def algebra():
    def flip():
        return Bernoulli(0.5).sample()
    def NUM():
        return Multinomial(range(5)).sample()
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
    return Multinomial(range(4)).sample()

def utterance_prior():
    return Multinomial([
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
    observe(Bernoulli(1.0), m)
    return world

@infer
def speaker(world):
    utterance = utterance_prior()
    L = literal_listener(utterance)
    observe(L, world)
    return utterance

@infer
def listener(utterance):
    world = world_prior()
    S = speaker(world)
    observe(S, utterance)
    return world

def test_scalar_implicature():
    assert listener('some of the people are nice').isclose(Multinomial(
        [1, 2, 3],
        probabilities=[4/9, 4/9, 1/9],
    ))

def test_builtins():
    @infer
    def model():
        return abs(Multinomial([-1, 0, 1]).sample())
    assert model().isclose(Multinomial(
        [0, 1],
        probabilities=[1/3, 2/3],
    ))
