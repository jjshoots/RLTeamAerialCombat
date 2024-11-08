import io
from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn
from pydantic import BaseModel, StrictStr

from dogfighter.models.mdp_types import Action, Observation


class ActorConfig(BaseModel):
    """ActorConfig for creating actors."""

    _registry: ClassVar[set[str]] = set()
    variant: StrictStr = "null"

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


class Actor(nn.Module):
    """A generic actor."""

    def save(self, target: str | Path | io.BytesIO) -> None:
        """save."""
        torch.save(self.state_dict(), target)

    def load(self, target: str | Path | io.BytesIO) -> None:
        """load."""
        self.load_state_dict(
            torch.load(
                target,
                map_location=torch.device(self.device),
                weights_only=True,
            )
        )

    def to(self, *args, **kwargs) -> "Actor":
        """Clears the internal device pointer and performs a conventional `to`."""
        self.__dict__.pop("device", None)
        return super().to(*args, **kwargs)

    @cached_property
    def device(self) -> torch.device:
        """device.

        Args:

        Returns:
            torch.device:
        """
        return next(self.parameters()).device

    def __call__(self, obs: Observation) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """__call__.

        Args:
            obs (Observation): obs

        Returns:
            torch.Tensor:
        """
        return self.forward(obs=obs)

    @abstractmethod
    def forward(self, obs: Observation) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """forward.

        Args:
            obs (Observation): obs

        Returns:
            torch.Tensor:
        """
        raise NotImplementedError

    @staticmethod
    def sample(*args, **kwargs) -> Action | tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @staticmethod
    def infer(*args, **kwargs) -> Action:
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


class Critic(nn.Module):
    """A generic critic."""

    def save(self, file_obj: io.BytesIO) -> None:
        """save."""
        torch.save(self.state_dict(), file_obj)

    def load(self, file_obj: io.BytesIO) -> None:
        """load."""
        self.load_state_dict(
            torch.load(
                file_obj,
                map_location=torch.device(self._device),
                weights_only=True,
            )
        )

    def to(self, *args, **kwargs) -> "Critic":
        """Clears the internal device pointer and performs a conventional `to`."""
        self.__dict__.pop("device", None)
        return super().to(*args, **kwargs)

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
