from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic

import torch
import torch.nn as nn
from pydantic import BaseModel, StrictBool
from wingman.replay_buffer import ReplayBuffer

from dogfighter.bases.base_types import Action, Observation


class AlgorithmConfig(BaseModel):
    """AlgorithmConfig for instantiating an algorithm"""

    compile: StrictBool

    @abstractmethod
    def instantiate(self) -> Algorithm:
        """instantiate.

        Args:

        Returns:
            Algorithm:
        """
        raise NotImplementedError


class Algorithm(nn.Module, Generic[Observation, Action]):
    """Algorithm."""

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
