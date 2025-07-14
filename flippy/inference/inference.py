from typing import Callable, Generic, TypeVar, Dict, Any, Tuple, List, Union, Optional
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from flippy.distributions import Distribution, Element
from flippy.distributions.builtin_dists import Categorical

MarginalLikelihood = float

class InferenceAlgorithm(ABC, Generic[Element]):
    @abstractmethod
    def run(self, *args, **kws) -> Distribution[Element]:
        raise NotImplementedError

class InferenceResult(ABC, Distribution[Element]):
    @property
    @abstractmethod
    def marginal_likelihood(self) -> float:
        raise NotImplementedError

class DiscreteInferenceResult(InferenceResult, Categorical[Element]):
    def __init__(self, support, probabilities, marginal_likelihood):
        Categorical.__init__(self, support=support, probabilities=probabilities)
        self._marginal_likelihood = marginal_likelihood

    @classmethod
    def from_values_scores(cls, return_values: List[Element], return_scores: List[float]):
        assert len(return_values) == len(return_scores)
        return_scores = np.array(return_scores)
        max_score = np.max(return_scores)
        return_probs = np.exp(return_scores - max_score)
        return_probs = return_probs / np.sum(return_probs)
        values_probs = defaultdict(float)
        for value, prob in zip(return_values, return_probs):
            values_probs[value] += prob
        values, probs = zip(*values_probs.items())
        log_marginal_likelihood = max_score + np.log(np.sum(np.exp(return_scores - max_score)))
        marginal_likelihood = np.exp(log_marginal_likelihood)
        return cls(
            support=values,
            probabilities=probs,
            marginal_likelihood=marginal_likelihood
        )

    @property
    def marginal_likelihood(self) -> float:
        return self._marginal_likelihood
