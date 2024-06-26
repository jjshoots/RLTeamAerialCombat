from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func
from pydantic import BaseModel


class EnvParams(BaseModel):
    pass


class ModelParams(BaseModel):
    pass


class AlgorithmParams(BaseModel):
    pass


@dataclass
class Observation:
    pass


ObservationType = TypeVar("ObservationType", bound=Observation)


Action = torch.Tensor


class BaseQUEnsemble(nn.Module, Generic[ObservationType]):
    def __init__(self, env_params: EnvParams, model_params: ModelParams) -> None:
        super().__init__()

    def __call__(self, obs: ObservationType, act: Action) -> torch.Tensor:
        return self.forward(obs=obs, act=act)

    @abstractmethod
    def forward(self, obs: ObservationType, act: Action) -> torch.Tensor:
        raise NotImplementedError


class BaseActor(nn.Module, Generic[ObservationType]):
    def __init__(self, env_params: EnvParams, model_params: ModelParams) -> None:
        super().__init__()

    def __call__(self, obs: ObservationType) -> torch.Tensor:
        return self.forward(obs=obs)

    @abstractmethod
    def forward(self, obs: ObservationType) -> torch.Tensor:
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
