import torch
import torch.nn as nn
from wingman import NeuralBlocks

from dogfighter.models.bases import Action, BaseCritic
from dogfighter.models.transformer.transformer_backbone import \
    TransformerBackbone
from dogfighter.models.transformer.transformer_bases import (
    TransformerEnvParams, TransformerModelParams, TransformerObservation)


class TransformerQUNetwork(nn.Module, BaseCritic[TransformerObservation]):
    """A classic Q network that uses a transformer backbone."""

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
        super().__init__()

        # the basic backbone
        self.backbone = TransformerBackbone(
            env_params=env_params,
            model_params=model_params,
        )

        # the network to get the action representation
        _features_description = [env_params.act_size, model_params.embed_dim]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.act_network = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # the network to merge the action and obs/att representations
        _features_description = [2 * model_params.embed_dim, 2]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.head = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # register the bias for the uncertainty
        self.register_buffer(
            "uncertainty_bias", torch.tensor(1) * 999.9, persistent=True
        )

    @torch.jit.script
    def forward(
        self,
        obs: TransformerObservation,
        act: Action,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (TransformerObservation): obs
            act (Action): Action of shape [B, act_size] or [num_actions, B, act_size]

        Returns:
            torch.Tensor: Q value and Uncertainty tensor of shape [q_u, B] or [q_u, num_actions, B]
        """
        # pass obs and att through the backbone
        # the shape here is [B, N, embed_dim]
        obsatt_embed = self.backbone(obs=obs.obs, obs_mask=obs.obs_mask, att=obs.att)

        # pass the action through the action network
        # the shape here is either [B, embed_dim] or [num_actions, B, embed_dim]
        action_embed = self.act_network(act)

        # if we have multiple actions per observation, stack the observation
        if len(act.shape) != len(obs.att.shape):
            obsatt_embed = obsatt_embed.expand(act.shape[0], -1, -1)

        # merge things together and get the output
        # the shape here is either [B, q_u] or [num_actions, B, q_u]
        q_u = self.head(torch.cat([obsatt_embed, action_embed], dim=-1))

        # move the qu to the first dim
        # the shape here is either [q_u, B] or [q_u, num_actions, B]
        q_u = torch.movedim(q_u, 0, -1)

        # return Q and U
        return q_u
