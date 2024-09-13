from dataclasses import field
from typing import Literal

import torch
from pydantic import StrictInt
from torch import nn

from dogfighter.models.critics import (
    UncertaintyAwareCritic,
    UncertaintyAwareCriticConfig,
)


class MlpQUNetworkConfig(UncertaintyAwareCriticConfig):
    """MlpQUNetworkConfig."""

    variant: Literal["mlp"] = "mlp"  # pyright: ignore
    obs_size: StrictInt
    act_size: StrictInt
    embed_dim: StrictInt = field(default=256)

    def instantiate(self) -> "MlpQUNetwork":
        return MlpQUNetwork(self)


class MlpQUNetwork(UncertaintyAwareCritic):
    """A classic Q network that uses a transformer backbone."""

    def __init__(self, config: MlpQUNetworkConfig) -> None:
        """__init__.

        Args:
            config (MlpQUNetworkConfig): config

        Returns:
            None:
        """
        super().__init__()

        # outputs the action after all the compute before it
        self.head = nn.Sequential(
            nn.Linear(config.obs_size + config.act_size, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 2),
        )

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (torch.Tensor): Observation of shape [B, obs_size]
            act (torch.Tensor): Action of shape [B, act_size] or [num_actions, B, act_size]

        Returns:
            torch.Tensor: Q value and Uncertainty tensor of shape [q_u, B] or [q_u, num_actions, B]
        """

        # if we have multiple actions per observation, stack the observation
        if len(act.shape) != len(obs.shape):
            input = obs.expand(act.shape[0], -1, -1)
        else:
            input = obs

        # get the output
        # the shape here is either [B, q_u] or [num_actions, B, q_u]
        q_u = self.head(torch.cat([input, act], dim=-1))

        # move the qu to the first dim
        # the shape here is either [q_u, B] or [q_u, num_actions, B]
        q_u = torch.movedim(q_u, 0, -1)

        # return Q and U
        return q_u
