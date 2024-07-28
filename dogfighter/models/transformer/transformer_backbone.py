from typing import Literal

import torch
import torch.nn as nn
from wingman import NeuralBlocks


class TransformerBackbone(nn.Module):
    """This takes the observation and attitude and converts it into an embedding representation."""

    def __init__(
        self,
        input_dim: int,
        ff_dim: int,
        num_att_heads: int,
        num_decode_layers: int,
        num_encode_layers: int,
    ) -> None:
        """__init__.

        Input SRC and TGT are expected to be [B, N, obs_size] tensors.

        Args:
            input_dim (int): input_dim
            ff_dim (int): ff_dim
            num_att_heads (int): num_att_heads
            num_decode_layers (int): num_decode_layers
            num_encode_layers (int): num_encode_layers

        Returns:
            None:
        """
        super().__init__()

        # self attention -> d_model
        _features_description = [
            input_dim,
            input_dim,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.obs_embedder = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # cross attention -> d_model
        _features_description = [
            input_dim,
            input_dim,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.att_embedder = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # the transformer model
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=num_att_heads,
            num_encoder_layers=num_encode_layers,
            num_decoder_layers=num_decode_layers,
            dim_feedforward=ff_dim,
            batch_first=True,
        )

    def forward(
        self, obs: dict[Literal["src", "tgt", "mask"], torch.Tensor]
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (dict[Literal["src", "tgt", "mask"], torch.Tensor]): [B, N, obs_size] tensors

        Returns:
            torch.Tensor:
        """
        # TODO: handle observation mask

        # this is [B, N, embed_dim] and [B, 1, embed_dim]
        obs_embeddings = self.obs_embedder(obs["src"])
        att_embeddings = self.att_embedder(obs["tgt"].unsqueeze(-2))

        # this is [B, 1, embed_dim] then [B, embed_dim]
        result = self.transformer(src=obs_embeddings, tgt=att_embeddings)
        result = result.squeeze(-2)
        return result
