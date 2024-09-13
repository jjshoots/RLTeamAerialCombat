from typing import Literal

import torch
from pydantic import StrictInt
from torch import nn

from dogfighter.models.critics import (
    UncertaintyAwareCritic,
    UncertaintyAwareCriticConfig,
)
from dogfighter.models.mlp.blocks.simba_block import SimbaBlock


class SimbaQUNetworkConfig(UncertaintyAwareCriticConfig):
    """SimbaQUNetworkConfig."""

    variant: Literal["simba"] = "simba"  # pyright: ignore
    obs_size: StrictInt
    act_size: StrictInt
    embed_dim: StrictInt
    num_blocks: StrictInt

    def instantiate(self) -> "SimbaQUNetwork":
        return SimbaQUNetwork(self)


class SimbaQUNetwork(UncertaintyAwareCritic):
    """A classic Q network using Simba as the core module."""

    def __init__(self, config: SimbaQUNetworkConfig) -> None:
        """__init__.

        Args:
            config (SimbaQUNetworkConfig): config

        Returns:
            None:
        """
        super().__init__()

        self.input_network = nn.Linear(
            config.obs_size + config.act_size, config.embed_dim
        )
        self.simba_network = SimbaBlock(config.embed_dim, config.num_blocks)
        self.output_network = nn.Linear(config.embed_dim, 2)

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
        q_u = self.output_network(
            self.simba_network(self.input_network(torch.cat([input, act], dim=-1)))
        )

        # move the qu to the first dim
        # the shape here is either [q_u, B] or [q_u, num_actions, B]
        q_u = torch.movedim(q_u, 0, -1)

        # return Q and U
        return q_u
