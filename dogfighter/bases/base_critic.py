from __future__ import annotations

from abc import abstractmethod
from typing import Generic

import torch
import torch.nn as nn
from pydantic import BaseModel

from dogfighter.bases.base_types import Action, Observation


class QUNetworkConfig(BaseModel):
    """QUNetworkConfig for creating QU Networks."""

    @abstractmethod
    def instantiate(self) -> QUNetwork:
        """instantiate.

        Args:

        Returns:
            QUNetwork:
        """
        raise NotImplementedError


class QUNetwork(nn.Module, Generic[Observation, Action]):
    """QUNetwork."""

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
