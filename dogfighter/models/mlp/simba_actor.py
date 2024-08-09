from dataclasses import field

import torch
from torch import nn

from dogfighter.bases.base_actor import Actor, ActorConfig
from dogfighter.models.mlp.blocks.simba_block import SimbaBlock


class SimbaActorConfig(ActorConfig):
    """SimbaActorConfig."""

    obs_size: int
    act_size: int
    embed_dim: int
    num_blocks: int

    def instantiate(self) -> "SimbaActor":
        return SimbaActor(self)


class SimbaActor(Actor):
    """Actor with Gaussian prediction head."""

    def __init__(self, config: SimbaActorConfig) -> None:
        """__init__.

        Args:
            config (SimbaActorConfig): config

        Returns:
            None:
        """
        super().__init__()

        self.input_network = nn.Linear(config.obs_size, config.embed_dim)
        self.simba_network = SimbaBlock(config.embed_dim, config.num_blocks)
        self.output_network = nn.Linear(config.embed_dim, config.act_size * 2)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (torch.Tensor): observation of shape [B, obs_size]

        Returns:
            torch.Tensor:
        """
        # output here is shape [B, act_size * 2]
        output = self.output_network(self.simba_network(self.input_network(obs)))

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
