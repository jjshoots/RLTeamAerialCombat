import numpy as np
import torch
from gymnasium.vector import VectorEnv, VectorEnvWrapper
from wingman.replay_buffer.core import Any
from wingman.utils import cpuize, gpuize


class VecEnvGpuizeWrapper(VectorEnvWrapper):
    """VecEnvGpuizeWrapper.
    """

    def __init__(self, env: VectorEnv, device: torch.device):
        """__init__.

        Args:
            env (VectorEnv): env
            device (torch.device): device
        """
        super().__init__(env)
        self.device = device

    def reset_wait(
        self,
        **kwargs,
    ) -> tuple[Any, dict[str, Any]]:
        """reset_wait.

        Args:
            kwargs:

        Returns:
            tuple[Any, dict[str, Any]]:
        """
        result: tuple[np.ndarray, dict[str, Any]] = super().reset_wait(**kwargs)
        return gpuize(result[0], device=self.device), result[1]

    def step_async(self, actions: torch.Tensor) -> None:
        """step_async.

        Args:
            actions (torch.Tensor): actions

        Returns:
            None:
        """
        np_actions = cpuize(actions)
        return super().step_async(np_actions)

    def step_wait(
        self,
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """step_wait.

        Args:

        Returns:
            tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """
        result: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
        ] = super().step_wait()
        obs = gpuize(result[0], device=self.device)

        return obs, *result[1:]
