from typing import Literal

import torch
from memorial import ReplayBuffer
from memorial.replay_buffers import FlatReplayBuffer
from memorial.wrappers import DictReplayBufferWrapper
from pydantic import BaseModel, StrictBool, StrictInt, StrictStr


class ReplayBufferConfig(BaseModel):
    """Replay Buffer Configuration.

    This just returns a Replay Buffer.
    """

    mem_size: StrictInt
    mode: Literal["torch", "numpy"]
    device: StrictStr
    use_dict_wrapper: StrictBool = False
    store_on_device: StrictBool = True
    random_rollover: StrictBool = True

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
