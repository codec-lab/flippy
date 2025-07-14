from flippy.inference.inference import InferenceAlgorithm
from flippy.inference.simpleenumeration import SimpleEnumeration
from flippy.inference.enumeration import Enumeration
from flippy.inference.sample_prior import SamplePrior
from flippy.inference.likelihood_weighting import LikelihoodWeighting
from flippy.inference.mcmc.metropolis_hastings import MetropolisHastings
from flippy.inference.max_marg_post import MaximumMarginalAPosteriori
from flippy.distributions import Categorical

def _distribution_from_inference(dist):
    ele, probs = zip(*dist.items())
    return Categorical(ele, probabilities=probs)
