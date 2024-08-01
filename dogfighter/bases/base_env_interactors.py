from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from gymnasium import Env
from gymnasium.vector import VectorEnv
from pettingzoo import ParallelEnv
from wingman.replay_buffer import ReplayBuffer

from dogfighter.bases.base_actor import Actor

SupportedEnvTypes = ParallelEnv | VectorEnv | Env


@runtime_checkable
class CollectFunctionProtocol(Protocol):
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
