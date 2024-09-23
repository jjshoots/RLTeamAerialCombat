from typing import Literal, cast

import torch
from pydantic import StrictInt
from torch import nn

from dogfighter.models.critics import (
    UncertaintyAwareCritic,
    UncertaintyAwareCriticConfig,
)
from dogfighter.models.mdp_types import Observation, TransformerObservation
from dogfighter.models.transformer.blocks.pre_ln_decoder import PreLNDecoder


class PreLNDecoderQUNetworkConfig(UncertaintyAwareCriticConfig):
    """PreLNDecoderQUNetworkConfig."""

    variant: Literal["pre_ln_decoder"] = "pre_ln_decoder"  # pyright: ignore
    src_size: StrictInt
    tgt_size: StrictInt
    act_size: StrictInt
    embed_dim: StrictInt
    ff_dim: StrictInt
    num_att_heads: StrictInt
    num_layers: StrictInt

    def instantiate(self) -> "PreLNDecoderQUNetwork":
        return PreLNDecoderQUNetwork(self)


class PreLNDecoderQUNetwork(UncertaintyAwareCritic):
    """A pre-layernorm decoder qu network."""

    def __init__(self, config: PreLNDecoderQUNetworkConfig) -> None:
        """__init__.

        Args:
            config (PreLNDecoderQUNetworkConfig): config

        Returns:
            None:
        """
        super().__init__()

        # network to go from src, tgt -> embed
        self.src_projection = nn.Linear(config.src_size, config.embed_dim)
        self.tgt_projection = nn.Linear(config.tgt_size, config.embed_dim)

        # pre layernorm decoder to go from embed -> embed
        self.decoders = nn.ModuleList(
            [
                PreLNDecoder(
                    embed_dim=config.embed_dim,
                    ff_dim=config.ff_dim,
                    num_heads=config.num_att_heads,
                )
                for _ in range(config.num_layers)
            ]
        )

        # network to get the action representation
        self.act_network = nn.Linear(config.act_size, config.embed_dim)

        # network to merge the action and obs/act representations
        self.head = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.ff_dim),
            nn.ReLU(),
            nn.Linear(config.ff_dim, 2),
        )

    def forward(
        self,
        obs: Observation,
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
        obs = cast(TransformerObservation, obs)

        # generate qkv tensors [B, N, embed_dim] for qkv
        q = self.tgt_projection(obs["tgt"])
        kv = self.src_projection(obs["src"])

        # pass the tensors into the decoder
        # the result here is [B, N, embed_dim]
        # take the last element over the tgt output
        for f in self.decoders:
            obs_embed = f(q=q, k=kv, v=kv, k_mask=obs["src_mask"].bool())
        obs_embed = obs_embed[..., -1, :]  # pyright: ignore[reportPossiblyUnboundVariable]

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
