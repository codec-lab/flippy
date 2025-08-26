from typing import Callable, Generic, TypeVar, Dict, Any, Tuple, List, Union, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
import math

from flippy.distributions import Distribution, Element
from flippy.distributions.builtin_dists import Categorical
from flippy.tools import PackagePlaceholder

try:
    import numpy as np
except ImportError:
    np = PackagePlaceholder("numpy")

MarginalLikelihood = float

class InferenceAlgorithm(ABC, Generic[Element]):
    @abstractmethod
    def run(self, *args, **kws) -> Distribution[Element]:
        raise NotImplementedError

    @property
    def is_cachable(self) -> bool:
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
    def from_values_scores(
        cls,
        return_values: List[Element],
        return_scores: List[float]
    ) -> 'DiscreteInferenceResult':
        assert len(return_values) == len(return_scores)
        try:
            return cls._from_values_scores_numpy(return_values, return_scores)
        except ImportError:
            return cls._from_values_scores_builtin(return_values, return_scores)

    @classmethod
    def _from_values_scores_numpy(
        cls,
        return_values: List[Element],
        return_scores: List[float]
    ) -> 'DiscreteInferenceResult':
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

    @classmethod
    def _from_values_scores_builtin(
        cls,
        return_values: List[Element],
        return_scores: List[float]
    ) -> 'DiscreteInferenceResult':
        max_score = max(return_scores)
        return_probs = [math.exp(score - max_score) for score in return_scores]
        total_prob = sum(return_probs)
        return_probs = [prob / total_prob for prob in return_probs]
        values_probs = defaultdict(float)
        for value, prob in zip(return_values, return_probs):
            values_probs[value] += prob
        values, probs = zip(*values_probs.items())
        log_marginal_likelihood = max_score + math.log(total_prob)
        marginal_likelihood = math.exp(log_marginal_likelihood)
        return cls(
            support=values,
            probabilities=probs,
            marginal_likelihood=marginal_likelihood
        )

    @property
    def marginal_likelihood(self) -> float:
        return self._marginal_likelihood
