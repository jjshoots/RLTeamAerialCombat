from typing import Any, ClassVar, Protocol, runtime_checkable

from gymnasium import Env
from gymnasium.vector import VectorEnv
from memorial import ReplayBuffer
from pettingzoo import ParallelEnv
from pydantic import BaseModel, StrictStr

from dogfighter.models.base import Actor

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
    _registry: ClassVar[set[str]] = set()
    variant: StrictStr

    def get_collection_fn(self) -> CollectionFunctionProtocol:
        raise NotImplementedError

    def get_evaluation_fn(self) -> EvaluationFunctionProtocol:
        raise NotImplementedError

    def get_display_fn(self) -> DisplayFunctionProtocol:
        raise NotImplementedError

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        variant = cls.__annotations__.get("variant")
        assert variant is not None
        if variant in cls._registry:
            raise ValueError(f"`variant` {variant} is already in use by another class.")
        cls._registry.add(variant)
