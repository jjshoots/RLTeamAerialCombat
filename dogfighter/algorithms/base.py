from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn
from memorial import ReplayBuffer
from pydantic import BaseModel, StrictBool, StrictStr

from dogfighter.models import KnownActorConfigs
from dogfighter.models.actors import GaussianActor


class AlgorithmConfig(BaseModel):
    """AlgorithmConfig for instantiating an algorithm"""

    _registry: ClassVar[set[str]] = set()
    variant: StrictStr
    compile: StrictBool
    actor_config: KnownActorConfigs
    device: StrictStr

    @abstractmethod
    def instantiate(self) -> "Algorithm":
        """instantiate.

        Args:

        Returns:
            Algorithm:
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


class Algorithm(nn.Module):
    """Algorithm."""

    @property
    def actor(self) -> GaussianActor:
        raise NotImplementedError

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

    @abstractmethod
    def update(
        self,
        memory: ReplayBuffer,
    ) -> dict[str, int | float | bool | str]:
        """update.

        Args:
            memory (ReplayBuffer): memory

        Returns:
            dict[str, Any]:
        """
        raise NotImplementedError
