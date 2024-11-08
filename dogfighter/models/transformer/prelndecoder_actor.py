from typing import Literal, cast

import torch
from pydantic import StrictInt
from torch import nn

from dogfighter.models.actors import GaussianActor, GaussianActorConfig
from dogfighter.models.mdp_types import Observation, TransformerObservation
from dogfighter.models.transformer.blocks.pre_ln_decoder import PreLNDecoder


class PreLNDecoderActorConfig(GaussianActorConfig):
    """PreLNDecoderActorConfig."""

    variant: Literal["pre_ln_decoder"] = "pre_ln_decoder"  # pyright: ignore
    src_size: StrictInt
    tgt_size: StrictInt
    act_size: StrictInt
    embed_dim: StrictInt
    ff_dim: StrictInt
    num_att_heads: StrictInt
    num_layers: StrictInt

    def instantiate(self) -> "PreLNDecoderActor":
        """instantiate.

        Args:

        Returns:
            "TransformerActor":
        """
        return PreLNDecoderActor(self)


class PreLNDecoderActor(GaussianActor):
    """A pre-layernorm decoder actor."""

    def __init__(self, config: PreLNDecoderActorConfig) -> None:
        """__init__.

        Args:
            config (PreLNDecoderActorConfig): config

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

        # outputs the action after all the compute before it
        self.head = nn.Linear(config.embed_dim, config.act_size * 2)

    def forward(
        self,
        obs: Observation,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (dict[Literal["src", "tgt", "src_mask", "tgt_mask"], torch.Tensor]):
                - "src": [batch_size, src_seq_len, obs_size] tensor, in most cases, this is number of other entities in the environment
                - "tgt": [batch_size, tgt_seq_len, obs_size] tensor, in most cases, this is 1
                - "src_mask": [batch_size, src_seq_len] tensor with False elements indicating unmasked positions
                - "tgt_mask": [batch_size, tgt_seq_len] tensor with False elements indicating unmasked positions

        Returns:
            torch.Tensor:
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

        # output here is shape [B, act_size * 2]
        output = self.head(obs_embed)

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
