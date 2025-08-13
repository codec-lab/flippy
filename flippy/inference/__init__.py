"""
Several inference algorithms are available depending on the use case.
For discrete models with small state spaces, exact inference can be performed
using enumeration. For larger discrete models and models with continuous variables,
approximate inference can be performed using sampling-based methods.
"""

from flippy.inference.inference import InferenceAlgorithm
from flippy.inference.simpleenumeration import SimpleEnumeration
from flippy.inference.enumeration import Enumeration
from flippy.inference.sample_prior import SamplePrior
from flippy.inference.likelihood_weighting import LikelihoodWeighting
from flippy.inference.mcmc.metropolis_hastings import MetropolisHastings
from flippy.inference.max_marg_post import MaximumMarginalAPosteriori
from flippy.distributions import Categorical

__all__ = [
    "SimpleEnumeration",
    "Enumeration",
    "SamplePrior",
    "LikelihoodWeighting",
    "MetropolisHastings",
    "MaximumMarginalAPosteriori",
]

def _distribution_from_inference(dist):
    ele, probs = zip(*dist.items())
    return Categorical(ele, probabilities=probs)
