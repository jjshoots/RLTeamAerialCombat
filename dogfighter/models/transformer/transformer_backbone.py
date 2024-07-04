import torch
import torch.nn as nn
from wingman import NeuralBlocks

from dogfighter.models.transformer.transformer_bases import (
    TransformerEnvParams, TransformerModelParams, TransformerObservation)


class TransformerBackbone(nn.Module):
    """This takes the observation and attitude and converts it into an embedding representation."""

    def __init__(
        self,
        env_params: TransformerEnvParams,
        model_params: TransformerModelParams,
    ) -> None:
        """A backbone model for encoding the observation and attitude of the current UAV.

        Observation is expected to be an [B, N, obs_size] tensor.
        Attitude is a [B, att_size,] tensor.

        Args:
            env_params (TransformerEnvParams): env_params
            model_params (TransformerModelParams): model_params

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

    def forward(self, obs: TransformerObservation) -> torch.Tensor:
        """forward.

        Args:
            obs (TransformerObservation):
                - "key": a [B, N, obs_size] tensor.
                - "mask": a [B, N, 1] mask tensor, where True means the key is null.
                - "query": a [B, att_size] tensor.

        Returns:-
            torch.Tensor: a [B, embed_dim] tensor representing the compressed state.
        """
        # TODO: handle observation mask

        # this is [B, N, embed_dim] and [B, 1, embed_dim]
        obs_embeddings = self.obs_embedder(obs["key"])
        att_embeddings = self.att_embedder(obs["query"].unsqueeze(-2))

        # this is [B, 1, embed_dim] then [B, embed_dim]
        result = self.transformer(src=obs_embeddings, tgt=att_embeddings)
        result = result.squeeze(-2)
        return result
