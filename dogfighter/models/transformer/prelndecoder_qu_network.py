from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from dogfighter.bases.base_critic import QUNetwork, QUNetworkConfig
from dogfighter.models.transformer.blocks.pre_ln_decoder import PreLNDecoder


class PreLNDecoderQUNetworkConfig(QUNetworkConfig):
    """PreLNDecoderQUNetworkConfig."""

    src_size: int
    tgt_size: int
    act_size: int
    embed_dim: int
    ff_dim: int
    num_tgt_context: int
    num_att_heads: int
    num_layers: int

    def instantiate(self) -> "PreLNDecoderQUNetwork":
        return PreLNDecoderQUNetwork(self)


class PreLNDecoderQUNetwork(QUNetwork):
    """A pre-layernorm decoder actor.

    The target is expanded from
    [batch_dim, seq_len, embed_dim]
    to
    [batch_dim, num_tgt_context * seq_len, embed_dim]
    to allow there to be more context vectors that the decoder can sample from.

    The output is then averaged over the num_tgt_context*seq_len dimension to form
    [batch_dim, embed_dim]
    before the final linear layer.
    """

    def __init__(self, config: PreLNDecoderQUNetworkConfig) -> None:
        """__init__.

        Args:
            config (PreLNDecoderQUNetworkConfig): config

        Returns:
            None:
        """
        super().__init__()

        # the pre layernorm decoder
        self.decoder = PreLNDecoder(
            dim_model=config.embed_dim,
            dim_feedforward=config.ff_dim,
            num_heads=config.num_att_heads,
            num_layers=config.num_layers,
        )

        # network to go from src, tgt -> embed
        self.src_network = nn.Linear(config.src_size, config.embed_dim)
        self.tgt_networks = nn.ModuleList(
            [
                nn.Linear(config.tgt_size, config.embed_dim)
                for _ in range(config.num_tgt_context)
            ]
        )

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
        # generate qkv tensors, tgt must be expanded to have more context
        # [B, N * num_context, embed_dim] for q
        # [B, N, embed_dim] for kv
        q = torch.concatenate([net(obs["tgt"]) for net in self.tgt_networks], dim=-2)
        kv = self.src_network(obs["src"])

        # pass the tensors into the decoder
        # the result here is [B, N * num_context, embed_dim]
        # take the mean over the second last dim
        obs_embed = self.decoder(q=q, k=kv, v=kv, k_mask=obs["src_mask"].bool())
        obs_embed = torch.mean(obs_embed, dim=-2)

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
