from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic

import torch.nn as nn
from pydantic import BaseModel
from wingman.replay_buffer import ReplayBuffer

from dogfighter.bases.base_types import Action, Observation


class AlgorithmConfig(BaseModel):
    """AlgorithmConfig for instantiating an algorithm"""

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
