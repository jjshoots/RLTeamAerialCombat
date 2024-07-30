from dataclasses import field

import torch
from wingman import NeuralBlocks

from dogfighter.bases.base_actor import Actor, ActorConfig


class MlpActorConfig(ActorConfig):
    """MlpActorConfig."""

    obs_size: int
    act_size: int
    embed_dim: int = field(default=256)

    def instantiate(self) -> "MlpActor":
        """instantiate.

        Args:

        Returns:
            Actor:
        """
        return MlpActor(self)


class MlpActor(Actor):
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
        _features_description = [
            config.obs_size,
            config.embed_dim,
            config.embed_dim,
            config.act_size * 2,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.head = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (torch.Tensor): Observation of shape [B, obs_size]

        Returns:
            torch.Tensor:
        """
        # output here is shape [B, act_size * 2]
        output = self.head(obs)

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
