from typing import Generic

from dogfighter.algorithms.base import Action, Observation
from dogfighter.models.base.base import Critic, CriticConfig


class UncertaintyAwareCriticConfig(CriticConfig):
    """UncertaintyAwareCriticConfig."""


class UncertaintyAwareCritic(Critic, Generic[Observation, Action]):
    """UncertaintyAwareCritic."""
