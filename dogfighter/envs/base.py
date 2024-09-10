import dataclasses
from abc import abstractmethod
from typing import Any

from gymnasium import Env
from gymnasium.vector import AsyncVectorEnv, VectorEnv
from pettingzoo import ParallelEnv
from pydantic import BaseModel


class SAEnvConfig(BaseModel):
    """Initializer for a single agent Gymnasium environment."""

    env_id: str
    render_mode: str | bool | None = None
    env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    @abstractmethod
    def instantiate(self) -> Env:
        raise NotImplementedError


class SAVecEnvConfig(BaseModel):
    """Initializer for single agent vector environments."""

    sa_env_config: SAEnvConfig
    num_envs: int

    def instantiate(self) -> VectorEnv:
        return AsyncVectorEnv(
            [lambda _=i: self.sa_env_config.instantiate() for i in range(self.num_envs)]
        )


class MAEnvConfig(BaseModel):
    """Initializer for a PettingZoo environment."""

    env_id: str
    render_mode: str | bool | None = None
    env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    @abstractmethod
    def instantiate(self) -> ParallelEnv:
        raise NotImplementedError
