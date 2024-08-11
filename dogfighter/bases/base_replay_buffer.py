from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel
from wingman.replay_buffer import FlatReplayBuffer, ReplayBuffer
from wingman.replay_buffer.wrappers import DictReplayBufferWrapper


class ReplayBufferConfig(BaseModel):
    """Replay Buffer Configuration.

    This just returns a FlatReplayBuffer.
    """

    mem_size: int
    mode: Literal["torch", "numpy"]
    device: str
    use_dict_wrapper: bool
    store_on_device: bool = True
    random_rollover: bool = True

    def instantiate(self) -> ReplayBuffer:
        """instantiate.

        Args:

        Returns:
            ReplayBuffer:
        """
        memory = FlatReplayBuffer(
            mem_size=self.mem_size,
            mode=self.mode,
            device=torch.device(self.device),
            store_on_device=self.store_on_device,
            random_rollover=self.random_rollover,
        )
        if self.use_dict_wrapper:
            memory = DictReplayBufferWrapper(memory)

        return memory
