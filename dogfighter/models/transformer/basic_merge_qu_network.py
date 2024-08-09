from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from dogfighter.bases.base_critic import QUNetwork, QUNetworkConfig


class BasicMergeQUNetworkConfig(QUNetworkConfig):
    """BasicMergeQUNetworkConfig."""

    src_size: int
    tgt_size: int
    act_size: int
    embed_dim: int

    def instantiate(self) -> "BasicMergeQUNetwork":
        return BasicMergeQUNetwork(self)


class BasicMergeQUNetwork(QUNetwork):
    """A classic Q network that uses a transformer backbone."""

    def __init__(self, config: BasicMergeQUNetworkConfig) -> None:
        """__init__.

        Args:
            config (BasicMergeQUNetworkConfig): config

        Returns:
            None:
        """
        super().__init__()

        # network to go from src, tgt -> embed
        self.src_network = nn.Linear(config.src_size, config.embed_dim)
        self.tgt_network = nn.Linear(config.tgt_size, config.embed_dim)

        # network to get the action representation
        self.act_network = nn.Linear(config.act_size, config.embed_dim)

        # network to merge the action and obs/att representations
        self.head = nn.Sequential(
            nn.Linear(3 * config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 2),
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
                - "src_mask": [batch_size, src_seq_len] tensor with False elements indicating unmasked positions
                - "tgt_mask": [batch_size, tgt_seq_len] tensor with False elements indicating unmasked positions
            act (torch.Tensor): Action of shape [B, act_size] or [num_actions, B, act_size]

        Returns:
            torch.Tensor: Q value and Uncertainty tensor of shape [q_u, B] or [q_u, num_actions, B]
        """
        # shape of src/tgt_embed is [batch_size, obs_size]
        # shape of act_embed is [batch_size, act_size] or [num_actions, batch_size, act_size]
        src_embed = torch.mean(
            self.src_network(obs["src"]) * obs["src_mask"][..., None].bool(), dim=-2
        )
        tgt_embed = torch.mean(
            self.tgt_network(obs["tgt"]) * obs["tgt_mask"][..., None].bool(), dim=-2
        )
        act_embed = self.act_network(act)

        # pass the action through the action network
        # the shape here is either [B, embed_dim] or [num_actions, B, embed_dim]
        act_embed = self.act_network(act)

        # if we have multiple actions per observation, stack the observation
        if len(act.shape) != len(src_embed.shape):
            src_embed = src_embed.expand(act.shape[0], -1, -1)
            tgt_embed = tgt_embed.expand(act.shape[0], -1, -1)

        # merge things together and get the output
        # the shape here is either [B, q_u] or [num_actions, B, q_u]
        q_u = self.head(torch.cat([src_embed, tgt_embed, act_embed], dim=-1))

        # move the qu to the first dim
        # the shape here is either [q_u, B] or [q_u, num_actions, B]
        q_u = torch.movedim(q_u, 0, -1)

        # return Q and U
        return q_u
