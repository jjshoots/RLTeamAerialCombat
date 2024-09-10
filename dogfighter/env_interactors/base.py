from typing import Any, Protocol, runtime_checkable

from gymnasium import Env
from gymnasium.vector import VectorEnv
from pettingzoo import ParallelEnv
from pydantic import BaseModel
from memorial import ReplayBuffer

from dogfighter.models.base.base_actor import Actor

SupportedEnvTypes = ParallelEnv | VectorEnv | Env


@runtime_checkable
class CollectionFunctionProtocol(Protocol):
    def __call__(
        self,
        *,
        actor: Actor,
        env: SupportedEnvTypes,
        memory: ReplayBuffer,
        num_transitions: int,
        use_random_actions: bool,
    ) -> tuple[ReplayBuffer, dict[str, Any]]:
        raise NotImplementedError


@runtime_checkable
class EvaluationFunctionProtocol(Protocol):
    def __call__(
        self,
        *,
        actor: Actor,
        env: SupportedEnvTypes,
        num_episodes: int,
    ) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError


@runtime_checkable
class DisplayFunctionProtocol(Protocol):
    def __call__(
        self,
        *,
        actor: Actor,
        env: ParallelEnv | Env,
    ) -> None:
        raise NotImplementedError


class EnvInteractorConfig(BaseModel):
    def get_collection_fn(self) -> CollectionFunctionProtocol:
        raise NotImplementedError

    def get_evaluation_fn(self) -> EvaluationFunctionProtocol:
        raise NotImplementedError

    def get_display_fn(self) -> DisplayFunctionProtocol:
        raise NotImplementedError
