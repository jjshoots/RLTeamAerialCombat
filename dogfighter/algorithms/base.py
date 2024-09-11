from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn
from memorial import ReplayBuffer
from pydantic import BaseModel, StrictBool, StrictStr

from dogfighter.models import KnownActorConfigs
from dogfighter.models.base.base_actor import Actor

Observation = TypeVar("Observation")
Action = TypeVar("Action")


class AlgorithmConfig(BaseModel):
    """AlgorithmConfig for instantiating an algorithm"""

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


class Algorithm(nn.Module, Generic[Observation, Action]):
    """Algorithm."""

    @property
    def actor(self) -> Actor:
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
    ) -> dict[str, Any]:
        """update.

        Args:
            memory (ReplayBuffer): memory

        Returns:
            dict[str, Any]:
        """
        raise NotImplementedError
