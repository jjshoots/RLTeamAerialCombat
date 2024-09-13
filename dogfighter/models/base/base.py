from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Generic

import torch
import torch.nn as nn
from pydantic import BaseModel, StrictStr

from dogfighter.models.base import Action, Observation


class ActorConfig(BaseModel):
    """ActorConfig for creating actors."""

    _registry: ClassVar[set[str]] = set()
    variant: StrictStr = "hi"

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        variant = cls.__annotations__.get("variant")
        assert variant is not None
        if variant in cls._registry:
            raise ValueError(f"`variant` {variant} is already in use by another class.")
        cls._registry.add(variant)

    @abstractmethod
    def instantiate(self) -> "Actor":
        """instantiate.

        Args:

        Returns:
            BaseActor:
        """
        raise NotImplementedError


class Actor(nn.Module, Generic[Observation, Action]):
    def save(self, filepath: str | Path) -> None:
        """save."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str | Path) -> None:
        """load."""
        self.load_state_dict(
            torch.load(
                filepath,
                map_location=torch.device(self.device),
                weights_only=True,
            )
        )

    @cached_property
    def device(self) -> torch.device:
        """device.

        Args:

        Returns:
            torch.device:
        """
        return next(self.parameters()).device

    def __call__(self, obs: Observation) -> torch.Tensor:
        """__call__.

        Args:
            obs (Observation): obs

        Returns:
            torch.Tensor:
        """
        return self.forward(obs=obs)

    @abstractmethod
    def forward(self, obs: Observation) -> torch.Tensor:
        """forward.

        Args:
            obs (Observation): obs

        Returns:
            torch.Tensor:
        """
        raise NotImplementedError

    @staticmethod
    def sample(*args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @staticmethod
    def infer(*args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class CriticConfig(BaseModel):
    """CriticConfig for creating Critics."""

    _registry: ClassVar[set[str]] = set()
    variant: StrictStr

    @abstractmethod
    def instantiate(self) -> "Critic":
        """instantiate.

        Args:

        Returns:
            Critic:
        """
        raise NotImplementedError

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        variant = cls.__annotations__.get("variant")
        assert variant is not None
        if variant in cls._registry:
            raise ValueError(f"`variant` {variant} is already in use by another class.")
        cls._registry.add(variant)


class Critic(nn.Module, Generic[Observation, Action]):
    """Critic."""

    def save(self, filepath: str | Path) -> None:
        """save."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str | Path) -> None:
        """load."""
        self.load_state_dict(
            torch.load(
                filepath,
                map_location=torch.device(self._device),
                weights_only=True,
            )
        )

    @cached_property
    def device(self) -> torch.device:
        """device.

        Args:

        Returns:
            torch.device:
        """
        return next(self.parameters()).device

    def __call__(self, obs: Observation, act: Action) -> torch.Tensor:
        """__call__.

        Args:
            obs (Observation): obs
            act (Action): act

        Returns:
            torch.Tensor:
        """
        return self.forward(obs=obs, act=act)

    @abstractmethod
    def forward(self, obs: Observation, act: Action) -> torch.Tensor:
        """forward.

        Args:
            obs (Observation): obs
            act (Action): act

        Returns:
            torch.Tensor:
        """
        raise NotImplementedError
