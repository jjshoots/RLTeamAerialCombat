from typing import Generic

import torch
import torch.distributions as dist
import torch.nn.functional as func

from dogfighter.algorithms.base import Action, Observation
from dogfighter.models.base.base import Actor, ActorConfig


class GaussianActorConfig(ActorConfig):
    """GaussianActorConfig."""


class GaussianActor(Actor, Generic[Observation, Action]):
    """GaussianActor."""

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
