from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from dogfighter.bases.base_critic import QUNetwork, QUNetworkConfig


class TransformerQUNetworkConfig(QUNetworkConfig):
    """TransformerQUNetworkConfig."""

    src_size: int
    tgt_size: int
    act_size: int
    embed_dim: int
    ff_dim: int
    num_att_heads: int
    num_decode_layers: int
    num_encode_layers: int

    def instantiate(self) -> "TransformerQUNetwork":
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

        # network to go from src, tgt -> embed
        self.src_network = nn.Linear(config.src_size, config.embed_dim)
        self.tgt_network = nn.Linear(config.tgt_size, config.embed_dim)

        # network to get the action representation
        self.act_network = nn.Linear(config.act_size, config.embed_dim)

        # network to merge the action and obs/att representations
        self.head = nn.Linear(2 * config.embed_dim, 2)

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
                - "src_mask": [batch_size, src_seq_len] tensor with False elements indicating unmasked positions
                - "tgt_mask": [batch_size, tgt_seq_len] tensor with False elements indicating unmasked positions
            act (torch.Tensor): Action of shape [B, act_size] or [num_actions, B, act_size]

        Returns:
            torch.Tensor: Q value and Uncertainty tensor of shape [q_u, B] or [q_u, num_actions, B]
        """
        # pass the tensors into the transformer
        # the resultl here is [B, N, embed_dim], where we extract [B, -1, embed_dim]
        obs_embed = self.transformer(
            src=self.src_network(obs["src"]),
            tgt=self.tgt_network(obs["tgt"]),
            src_key_padding_mask=obs["src_mask"].bool(),
            tgt_key_padding_mask=obs["tgt_mask"].bool(),
        )[:, -1, :]

        # pass the action through the action network
        # the shape here is either [B, embed_dim] or [num_actions, B, embed_dim]
        act_embed = self.act_network(act)

        # if we have multiple actions per observation, stack the observation
        if len(act.shape) != len(obs_embed.shape):
            obs_embed = obs_embed.expand(act.shape[0], -1, -1)

        # merge things together and get the output
        # the shape here is either [B, q_u] or [num_actions, B, q_u]
        q_u = self.head(torch.cat([obs_embed, act_embed], dim=-1))

        # move the qu to the first dim
        # the shape here is either [q_u, B] or [q_u, num_actions, B]
        q_u = torch.movedim(q_u, 0, -1)

        # return Q and U
        return q_u
