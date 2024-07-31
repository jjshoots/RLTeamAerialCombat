from __future__ import annotations

from typing import Literal

import torch
from wingman import NeuralBlocks

from dogfighter.bases.base_critic import QUNetwork, QUNetworkConfig


class TransformerQUNetworkConfig(QUNetworkConfig):
    """TransformerQUNetworkConfig."""

    src_size: int
    tgt_size: int
    act_size: int
    embed_dim: int
    ff_dim: int
    num_att_heads: int
    src_input_dim: int
    tgt_input_dim: int
    num_decode_layers: int
    num_encode_layers: int

    def instantiate(self) -> "TransformerQUNetwork":
        """instantiate.

        Args:

        Returns:
            "TransformerQUNetwork":
        """
        return TransformerQUNetwork(self)


class TransformerQUNetwork(QUNetwork):
    """A classic Q network that uses a transformer backbone."""

    def __init__(self, config: TransformerQUNetworkConfig) -> None:
        """__init__.

        Args:
            config (TransformerQUNetworkConfig): config

        Returns:
            None:
        """
        super().__init__()

        # the transformer model
        self.transformer = torch.nn.Transformer(
            d_model=config.embed_dim,
            nhead=config.num_att_heads,
            num_encoder_layers=config.num_encode_layers,
            num_decoder_layers=config.num_decode_layers,
            dim_feedforward=config.ff_dim,
            batch_first=True,
        )

        # network to go from src -> embed
        _features_description = [config.src_size, config.embed_dim]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.src_input_network = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # network to go from tgt -> embed
        _features_description = [config.tgt_size, config.embed_dim]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.tgt_input_network = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # network to get the action representation
        _features_description = [config.act_size, config.embed_dim]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.act_network = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # network to merge the action and obs/att representations
        _features_description = [2 * config.embed_dim, 2]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.embed_act_merger = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # register the bias for the uncertainty
        self.register_buffer(
            "uncertainty_bias", torch.tensor(1) * 999.9, persistent=True
        )

    def forward(
        self,
        obs: dict[Literal["src", "tgt", "src_mask", "tgt_mask"], torch.Tensor],
        act: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (dict[Literal["src", "tgt", "src_mask", "tgt_mask"], torch.Tensor]):
                - "src": [batch_size, src_seq_len, obs_size] tensor
                - "tgt": [batch_size, tgt_seq_len, obs_size] tensor
                - "src_mask": [batch_size, src_seq_len] tensor with True elements indicating unmasked positions
                - "tgt_mask": [batch_size, tgt_seq_len] tensor with True elements indicating unmasked positions
            act (torch.Tensor): Action of shape [B, act_size] or [num_actions, B, act_size]

        Returns:
            torch.Tensor: Q value and Uncertainty tensor of shape [q_u, B] or [q_u, num_actions, B]
        """
        # pass the tensors into the transformer
        # the resultl here is [B, N, embed_dim], where we extract [B, -1, embed_dim]
        obs_embed = self.transformer(
            src=self.src_input_network(obs["src"]),
            tgt=self.tgt_input_network(obs["tgt"]),
            src_key_padding_mask=obs["src_mask"].logical_not(),
            tgt_key_padding_mask=obs["tgt_mask"].logical_not(),
        )[:, -1, :]

        # pass the action through the action network
        # the shape here is either [B, embed_dim] or [num_actions, B, embed_dim]
        act_embed = self.act_network(act)

        # if we have multiple actions per observation, stack the observation
        if len(act.shape) != len(obs["src"].shape):
            obs_embed = obs_embed.expand(act.shape[0], -1, -1)

        # merge things together and get the output
        # the shape here is either [B, q_u] or [num_actions, B, q_u]
        q_u = self.head(torch.cat([obs_embed, act_embed], dim=-1))

        # move the qu to the first dim
        # the shape here is either [q_u, B] or [q_u, num_actions, B]
        q_u = torch.movedim(q_u, 0, -1)

        # return Q and U
        return q_u
