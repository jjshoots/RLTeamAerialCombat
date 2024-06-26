import torch
from wingman import NeuralBlocks

from dogfighter.models.bases import BaseActor
from dogfighter.models.mlp.mlp_bases import (MlpEnvParams, MlpModelParams,
                                             MlpObservation)


class MlpActor(BaseActor[MlpObservation]):
    """Actor with Gaussian prediction head."""

    def __init__(
        self,
        env_params: MlpEnvParams,
        model_params: MlpModelParams,
    ) -> None:
        """__init__.

        Args:
            env_params (MlpEnvParams): env_params
            model_params (MlpModelParams): model_params

        Returns:
            None:
        """
        super().__init__(env_params=env_params, model_params=model_params)

        # outputs the action after all the compute before it
        _features_description = [
            env_params.obs_size,
            model_params.embed_dim,
            model_params.embed_dim,
            env_params.act_size * 2,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.head = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(
        self,
        obs: MlpObservation,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (MlpObservation): Observation of shape [B, obs_size]

        Returns:
            torch.Tensor:
        """
        # output here is shape [B, act_size * 2]
        output = self.head(obs.obs)

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
