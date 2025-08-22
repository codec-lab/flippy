from flippy import infer, flip, mem, condition, draw_from, uniform, normal
from flippy.distributions.builtin_dists import Bernoulli, Categorical, Multinomial, \
    Gaussian, Normal, Gamma, Uniform, Beta, Binomial, Geometric, Poisson, \
    BetaBinomial, Dirichlet, DirichletMultinomial, Mixture

def test_distributions():
    flip()
    draw_from(3)
    uniform()
    normal()

    for dist in [
        Bernoulli(0.5), Uniform(0, 1), Beta(2, 5), Binomial(trials=10, p=0.5),
        Categorical((1, 2, 3)),
        Multinomial(
            categorical_support=("A", "B", "C"),
            trials=10,
            probabilities=(0.2, 0.5, 0.3)
        ),
        Gaussian(0, 1), Normal(0, 1), Gamma(2, 2), Geometric(0.5),
        Poisson(rate=3), BetaBinomial(trials=2, alpha=1.2, beta=3.3),
        Dirichlet([0.2, 0.5, 0.3]),
        DirichletMultinomial(
            trials=10,
            alphas=[0.2, 0.5, 0.3]
        ),
        Mixture(
            distributions=[
                Bernoulli(0.5),
                Gaussian(0, 1),
            ],
            weights=[0.5, 0.5]
        )
    ]:
        print(dist)
        x = dist.sample()
        assert dist.prob(x) > 0

def test_Enumeration():
    @infer(method="Enumeration")
    def simple_model():
        x = flip(0.5)
        y = flip(0.5)
        condition(x >= y)
        return x + y

    result = simple_model()
    assert dict(result) == {2: 1/3, 1: 1/3, 0: 1/3}

def test_MetropolisHastings():
    @infer(method="MetropolisHastings", samples=1000)
    def simple_model_mh():
        p = uniform()
        x = flip(p)
        y = flip(p)
        condition(x >= y)
        return x + y
    result_mh = simple_model_mh()

def test_mem():
    @mem
    def g(i):
        return flip()

    @infer
    def mem_model():
        return g(1) + g(1)

    result_mem = mem_model()
    assert dict(result_mem) == {2: 1/2, 0: 1/2}
