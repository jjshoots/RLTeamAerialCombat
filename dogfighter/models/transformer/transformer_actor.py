from typing import Literal, cast

import torch
from pydantic import StrictInt
from torch import nn

from dogfighter.models.actors import GaussianActor, GaussianActorConfig
from dogfighter.models.mdp_types import Observation, TransformerObservation


class TransformerActorConfig(GaussianActorConfig):
    """TransformerActorConfig."""

    variant: Literal["transformer"] = "transformer"  # pyright: ignore
    src_size: StrictInt
    tgt_size: StrictInt
    act_size: StrictInt
    embed_dim: StrictInt
    ff_dim: StrictInt
    num_att_heads: StrictInt
    num_encode_layers: StrictInt
    num_decode_layers: StrictInt

    def instantiate(self) -> "TransformerActor":
        return TransformerActor(self)


class TransformerActor(GaussianActor):
    """Actor with Gaussian prediction head."""

    def __init__(self, config: TransformerActorConfig) -> None:
        """__init__.

        Args:
            config (TransformerActorConfig): config

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

        # outputs the action after all the compute before it
        self.head = nn.Linear(config.embed_dim, config.act_size * 2)

    def forward(
        self,
        obs: Observation,
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
        obs = cast(TransformerObservation, obs)

        # pass the tensors into the transformer
        # the resultl here is [B, N, embed_dim], where we extract [B, -1, embed_dim]
        obs_embed = self.transformer(
            src=self.src_network(obs["src"]),
            tgt=self.tgt_network(obs["tgt"]),
            src_key_padding_mask=obs["src_mask"].bool(),
            tgt_key_padding_mask=obs["tgt_mask"].bool(),
        )[:, -1, :]

        # output here is shape [B, act_size * 2]
        output = self.head(obs_embed)

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
