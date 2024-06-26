import torch
from wingman import NeuralBlocks

from dogfighter.models.bases import BaseActor
from dogfighter.models.transformer.transformer_backbone import \
    TransformerBackbone
from dogfighter.models.transformer.transformer_bases import (
    TransformerEnvParams, TransformerModelParams, TransformerObservation)


class TransformerActor(BaseActor[TransformerObservation]):
    """Actor with Gaussian prediction head."""

    def __init__(
        self,
        env_params: TransformerEnvParams,
        model_params: TransformerModelParams,
    ) -> None:
        """__init__.

        Args:
            env_params (TransformerEnvParams): env_params
            model_params (TransformerModelParams): model_params

        Returns:
            None:
        """
        super().__init__(env_params=env_params, model_params=model_params)

        # the basic backbone
        self.backbone = TransformerBackbone(
            env_params=env_params,
            model_params=model_params,
        )

        # outputs the action after all the compute before it
        _features_description = [model_params.embed_dim, env_params.act_size * 2]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.head = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(
        self,
        obs: TransformerObservation,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (TransformerObservation): obs

        Returns:
            torch.Tensor:
        """
        # embedding here is shape [B, embed_dim]
        embedding = self.backbone(obs=obs.obs, obs_mask=obs.obs_mask, att=obs.att)

        # output here is shape [B, act_size * 2]
        output = self.head(embedding)

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
