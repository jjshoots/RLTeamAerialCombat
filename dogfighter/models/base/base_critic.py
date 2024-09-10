from abc import abstractmethod
from functools import cached_property
from typing import Generic, TypeVar

import torch
import torch.nn as nn
from pydantic import BaseModel

Observation = TypeVar("Observation")
Action = TypeVar("Action")


class QUNetworkConfig(BaseModel):
    """QUNetworkConfig for creating QU Networks."""

    @abstractmethod
    def instantiate(self) -> "QUNetwork":
        """instantiate.

        Args:

        Returns:
            QUNetwork:
        """
        raise NotImplementedError


class QUNetwork(nn.Module, Generic[Observation, Action]):
    """QUNetwork."""

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
