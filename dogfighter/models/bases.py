from abc import abstractmethod
from typing import Generic, TypeVar

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func
from pydantic import BaseModel, Field, StrictFloat, StrictInt


class EnvParams(BaseModel):
    obs_size: StrictInt
    att_size: StrictInt
    act_size: StrictInt


class ModelParams(BaseModel):
    qu_num_ensemble: StrictInt = Field(2)
    embed_dim: StrictInt = Field(128)
    att_inner_dim: StrictInt = Field(256)
    att_num_heads: StrictInt = Field(2)
    att_num_encoder_layers: StrictInt = Field(2)
    att_num_decoder_layers: StrictInt = Field(2)


class LearningParams(BaseModel):
    learning_rate: StrictFloat = Field(0.003)
    alpha_learning_rate: StrictFloat = Field(0.01)
    target_entropy: None | StrictFloat = Field(None)
    discount_factor: StrictFloat = Field(0.99)
    update_ratio: StrictInt = Field(1)
    actor_update_ratio: StrictInt = Field(1)
    critic_update_ratio: StrictInt = Field(1)


class Observation:
    pass


ObservationType = TypeVar("ObservationType", bound=Observation)


Action = torch.Tensor


class BaseCritic(nn.Module, Generic[ObservationType]):
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

    @torch.jit.script
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

    @torch.jit.script
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
