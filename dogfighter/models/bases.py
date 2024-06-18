from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
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


class BaseCritic(ABC, Generic[ObservationType]):
    @abstractmethod
    def __init__(self, env_params: EnvParams, model_params: ModelParams) -> None:
        raise NotImplementedError

    def __call__(self, obs: ObservationType, act: Action) -> torch.Tensor:
        return self.forward(obs=obs, act=act)

    @abstractmethod
    def forward(self, obs: ObservationType, act: Action) -> torch.Tensor:
        raise NotImplementedError


class BaseActor(ABC, Generic[ObservationType]):
    @abstractmethod
    def __init__(self, env_params: EnvParams, model_params: ModelParams) -> None:
        raise NotImplementedError

    def __call__(self, obs: ObservationType) -> torch.Tensor:
        return self.forward(obs=obs)

    @abstractmethod
    def forward(self, obs: ObservationType) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def sample(mean: torch.Tensor, var: torch.Tensor) -> tuple[Action, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def infer(mean: torch.Tensor, var: torch.Tensor) -> Action:
        raise NotImplementedError
