from typing import Literal

from dogfighter.models.base import Critic, CriticConfig


class UncertaintyAwareCriticConfig(CriticConfig):
    """UncertaintyAwareCriticConfig."""

    variant: Literal["qu_critic"] = "qu_critic"  # pyright: ignore


class UncertaintyAwareCritic(Critic):
    """UncertaintyAwareCritic."""
