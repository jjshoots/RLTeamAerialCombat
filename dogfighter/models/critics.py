from typing import Generic, Literal

from dogfighter.models.base import Critic, CriticConfig
from dogfighter.models.mdp_types import Action, Observation


class UncertaintyAwareCriticConfig(CriticConfig):
    """UncertaintyAwareCriticConfig."""

    variant: Literal["qu_critic"] = "qu_critic"  # pyright: ignore


class UncertaintyAwareCritic(Critic, Generic[Observation, Action]):
    """UncertaintyAwareCritic."""
