from dataclasses import field
from typing import Literal, cast

import torch
from pydantic import StrictInt
from torch import nn

from dogfighter.models.actors import GaussianActor, GaussianActorConfig
from dogfighter.models.mdp_types import MlpObservation, Observation


class MlpActorConfig(GaussianActorConfig):
    """MlpActorConfig."""

    variant: Literal["mlp"] = "mlp"  # pyright: ignore
    obs_size: StrictInt
    act_size: StrictInt
    embed_dim: StrictInt = field(default=256)

    def instantiate(self) -> "MlpActor":
        """instantiate.

        Args:

        Returns:
            Actor:
        """
        return MlpActor(self)


class MlpActor(GaussianActor):
    """Actor with Gaussian prediction head."""

    def __init__(self, config: MlpActorConfig) -> None:
        """__init__.

        Args:
            config (MlpActorConfig): config

        Returns:
            None:
        """
        super().__init__()

        # outputs the action after all the compute before it
        self.head = nn.Sequential(
            nn.Linear(config.obs_size, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.act_size * 2),
        )

    def forward(
        self,
        obs: Observation,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (torch.Tensor): observation of shape [B, obs_size]

        Returns:
            torch.Tensor:
        """
        obs = cast(MlpObservation, obs)

        # output here is shape [B, act_size * 2]
        output = self.head(obs)

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
