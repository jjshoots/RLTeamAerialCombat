from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from dogfighter.bases.base_actor import Actor, ActorConfig
from dogfighter.models.transformer.pre_ln_decoder import PreLNDecoder


class PreLNDecoderActorConfig(ActorConfig):
    """PreLNDecoderActorConfig."""

    src_size: int
    tgt_size: int
    act_size: int
    embed_dim: int
    ff_dim: int
    num_tgt_context: int
    num_att_heads: int
    num_layers: int

    def instantiate(self) -> "PreLNDecoderActor":
        """instantiate.

        Args:

        Returns:
            "TransformerActor":
        """
        return PreLNDecoderActor(self)


class PreLNDecoderActor(Actor):
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

    def __init__(self, config: PreLNDecoderActorConfig) -> None:
        """__init__.

        Args:
            config (PreLNDecoderActorConfig): config

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

        # outputs the action after all the compute before it
        self.head = nn.Linear(config.embed_dim, config.act_size * 2)

    def forward(
        self,
        obs: dict[Literal["src", "tgt", "src_mask", "tgt_mask"], torch.Tensor],
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (dict[Literal["src", "tgt", "src_mask", "tgt_mask"], torch.Tensor]):
                - "src": [batch_size, src_seq_len, obs_size] tensor
                - "tgt": [batch_size, tgt_seq_len, obs_size] tensor
                - "src_mask": [batch_size, src_seq_len] tensor with False elements indicating unmasked positions
                - "tgt_mask": [batch_size, tgt_seq_len] tensor with False elements indicating unmasked positions

        Returns:
            torch.Tensor:
        """
        # generate qkv tensors, tgt must be expanded to have more context
        # [B, N, embed_dim] for kv
        # [B, N * num_context, embed_dim] for q
        kv = self.src_network(obs["src"])
        q = torch.concatenate([net(obs["tgt"]) for net in self.tgt_networks], dim=-2)

        # pass the tensors into the decoder
        # the result here is [B, N * num_context, embed_dim]
        # take the mean over the second last dim
        obs_embed = self.decoder(q=q, k=kv, v=kv, k_mask=obs["src_mask"])
        obs_embed = torch.mean(obs_embed, dim=-2)

        # output here is shape [B, act_size * 2]
        output = self.head(obs_embed)

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
