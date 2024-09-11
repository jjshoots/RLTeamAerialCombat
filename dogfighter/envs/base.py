import dataclasses
from abc import abstractmethod
from typing import Any

from gymnasium import Env
from gymnasium.vector import AsyncVectorEnv, VectorEnv
from pettingzoo import ParallelEnv
from pydantic import BaseModel, StrictBool, StrictInt, StrictStr


class SAEnvConfig(BaseModel):
    """Initializer for a single agent Gymnasium environment."""

    env_id: StrictStr
    render_mode: StrictStr | StrictBool | None = None
    env_kwargs: dict[StrictStr, Any] = dataclasses.field(default_factory=dict)

    @abstractmethod
    def instantiate(self) -> Env:
        raise NotImplementedError


class SAVecEnvConfig(BaseModel):
    """Initializer for single agent vector environments."""

    sa_env_config: SAEnvConfig
    num_envs: StrictInt

    def instantiate(self) -> VectorEnv:
        return AsyncVectorEnv(
            [lambda _=i: self.sa_env_config.instantiate() for i in range(self.num_envs)]
        )


class MAEnvConfig(BaseModel):
    """Initializer for a PettingZoo environment."""

    env_id: StrictStr
    render_mode: StrictStr | StrictBool | None = None
    env_kwargs: dict[StrictStr, Any] = dataclasses.field(default_factory=dict)

    @abstractmethod
    def instantiate(self) -> ParallelEnv:
        raise NotImplementedError
