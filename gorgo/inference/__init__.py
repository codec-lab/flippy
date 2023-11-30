from gorgo.inference.enumeration import Enumeration, GraphEnumeration
from gorgo.inference.sample_prior import SamplePrior
from gorgo.inference.likelihood_weighting import LikelihoodWeighting
from gorgo.inference.mcmc.metropolis_hastings import MetropolisHastings
from gorgo.distributions import Categorical

def _distribution_from_inference(dist):
    ele, probs = zip(*dist.items())
    return Categorical(ele, probabilities=probs)
