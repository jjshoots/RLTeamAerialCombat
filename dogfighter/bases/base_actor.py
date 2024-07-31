from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import Generic

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func
from pydantic import BaseModel

from dogfighter.bases.base_types import Action, Observation


class ActorConfig(BaseModel):
    """ActorConfig for creating actors."""

    @abstractmethod
    def instantiate(self) -> Actor:
        """instantiate.

        Args:

        Returns:
            BaseActor:
        """
        raise NotImplementedError


class Actor(nn.Module, Generic[Observation, Action]):
    """Actor."""

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
    def sample(
        mean: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """sample.

        Args:
            mean (torch.Tensor): mean
            var (torch.Tensor): var

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
        """
        # lower bound sigma and bias it
        normals = dist.Normal(mean, func.softplus(var) + 1e-6)

        # sample from dist
        mu_samples = normals.rsample()
        actions = torch.tanh(mu_samples)

        # calculate log_probs
        log_probs = normals.log_prob(mu_samples) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return actions, log_probs

    @staticmethod
    def infer(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """infer.

        Args:
            mean (torch.Tensor): mean
            var (torch.Tensor): var

        Returns:
            torch.Tensor:
        """
        return torch.tanh(mean)
