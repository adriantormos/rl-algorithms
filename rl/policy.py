import abc
import random
from typing import Optional

import torch


class RLPolicy(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook_(cls, subclass):
        return hasattr(subclass, 'choose_action') and \
               callable(subclass.choose_action) and \
               hasattr(subclass, 'last_action_log_probability') and \
               callable(subclass.log_probability)

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def choose_action(self, state: torch.Tensor):
        pass

    def log_probability(self, a_probabilities: torch.Tensor, action) -> torch.Tensor:
        pass


class DiscreteEpsilonGreedyRLPolicy(RLPolicy):
    def __init__(self, epsilon: float, alpha: Optional[float] = None):
        self.steps = 0
        self.alpha = alpha
        self.epsilon = epsilon

    def choose_action(self, q_values: torch.Tensor):
        self.steps += 1
        x = random.random()
        if self.alpha is not None and x > max(1 / (self.alpha * self.steps), self.epsilon):
            return torch.tensor([q_values.max(dim=0)[1]], dtype=torch.int64)
        elif self.alpha is None and x > self.epsilon:
            return torch.tensor([q_values.max(dim=0)[1]], dtype=torch.int64)
        else:
            return torch.tensor([random.randrange(q_values.shape[0])], dtype=torch.int64)
