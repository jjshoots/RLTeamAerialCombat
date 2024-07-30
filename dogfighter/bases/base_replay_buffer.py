from __future__ import annotations

from typing import Literal, Union

import torch
from pydantic import BaseModel
from wingman.replay_buffer import FlatReplayBuffer, ReplayBuffer


class WrappedReplayBufferConfig(BaseModel):
    """Replay Buffer Wrapper Configuration.

    This contains the parameters of the replay buffer wrapper,
    as well as the configuration of the `base_buffer`,
    which can itself be a `ReplayBufferConfig` or another `WrappedReplayBufferConfig`.
    """

    base_buffer: Union[ReplayBufferConfig, "WrappedReplayBufferConfig"]

    def instantiate(self) -> ReplayBuffer:
        """instantiate.

        Args:

        Returns:
            ReplayBuffer:
        """
        raise NotImplementedError


class ReplayBufferConfig(BaseModel):
    """Replay Buffer Configuration.

    This just returns a FlatReplayBuffer.
    """

    mem_size: int
    mode: Literal["torch", "numpy"]
    device: str
    store_on_device: bool = True
    random_rollover: bool = True

    def instantiate(self) -> ReplayBuffer:
        """instantiate.

        Args:

        Returns:
            ReplayBuffer:
        """
        return FlatReplayBuffer(
            mem_size=self.mem_size,
            mode=self.mode,
            device=torch.device(self.device),
            store_on_device=self.store_on_device,
            random_rollover=self.random_rollover,
        )
