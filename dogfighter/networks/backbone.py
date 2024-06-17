import torch
import torch.nn as nn
from wingman import NeuralBlocks

from dogfighter.networks.dataclasses import EnvParams, ModelParams


class Backbone(nn.Module):
    """This takes the observation and attitude and converts it into an embedding representation."""

    def __init__(
        self,
        env_params: EnvParams,
        model_params: ModelParams,
    ) -> None:
        """A backbone model for encoding the observation and attitude of the current UAV.

        Observation is expected to be an [B, N, obs_size] tensor.
        Attitude is a [B, att_size,] tensor.

        Args:
            env_params (EnvParams): env_params
            model_params (ModelParams): model_params

        Returns:
            None:
        """
        super().__init__()

        # Observation -> d_model
        _features_description = [
            env_params.obs_size,
            model_params.embed_dim,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.obs_embedder = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # Attitude -> d_model
        _features_description = [
            env_params.att_size,
            model_params.embed_dim,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.att_embedder = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # the transformer model
        self.transformer = nn.Transformer(
            d_model=model_params.embed_dim,
            nhead=model_params.att_num_heads,
            num_encoder_layers=model_params.att_num_encoder_layers,
            num_decoder_layers=model_params.att_num_decoder_layers,
            dim_feedforward=model_params.att_inner_dim,
            batch_first=True,
        )

    @torch.jit.script
    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor, att: torch.Tensor
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (torch.Tensor): a [B, N, obs_size] tensor for N other UAVs.
            obs_mask (torch.Tensor): a [B, N, 1] mask tensor for N other UAVs, where True means the observation is null.
            obs (torch.Tensor): a [B, att_size] tensor for the current UAV's attitude.

        Returns:
            torch.Tensor: a [B, embed_dim] tensor representing the compressed state.
        """
        # TODO: handle observation mask

        # this is [B, N, embed_dim] and [B, 1, embed_dim]
        obs_embeddings = self.obs_embedder(obs)
        att_embeddings = self.att_embedder(att.unsqueeze(-2))

        # this is [B, 1, embed_dim] then [B, embed_dim]
        result = self.transformer(src=obs_embeddings, tgt=att_embeddings)
        result = result.squeeze(-2)
        return result
