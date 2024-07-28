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
        return MlpActor(
            obs_size=self.obs_size,
            act_size=self.act_size,
            embed_dim=self.embed_dim,
        )


class MlpActor(Actor):
    """Actor with Gaussian prediction head."""

    def __init__(
        self,
        obs_size: int,
        act_size: int,
        embed_dim: int,
    ) -> None:
        """__init__.

        Args:
            env_params (MlpEnvParams): env_params
            model_params (MlpModelParams): model_params

        Returns:
            None:
        """
        super().__init__()

        # outputs the action after all the compute before it
        _features_description = [
            obs_size,
            embed_dim,
            embed_dim,
            act_size * 2,
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
