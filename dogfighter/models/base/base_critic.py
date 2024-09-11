from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import torch
import torch.nn as nn
from pydantic import BaseModel, StrictStr

Observation = TypeVar("Observation")
Action = TypeVar("Action")


class QUNetworkConfig(BaseModel):
    """QUNetworkConfig for creating QU Networks."""

    _registry: ClassVar[set[str]] = set()
    variant: StrictStr

    @abstractmethod
    def instantiate(self) -> "QUNetwork":
        """instantiate.

        Args:

        Returns:
            QUNetwork:
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


class QUNetwork(nn.Module, Generic[Observation, Action]):
    """QUNetwork."""

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
