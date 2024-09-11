from typing import Literal

import torch
from pydantic import StrictInt
from torch import nn

from dogfighter.models.base.base_actor import Actor, ActorConfig


class BasicMergeActorConfig(ActorConfig):
    """BasicMergeActorConfig."""

    src_size: StrictInt
    tgt_size: StrictInt
    act_size: StrictInt
    embed_dim: StrictInt

    def instantiate(self) -> "BasicMergeActor":
        return BasicMergeActor(self)


class BasicMergeActor(Actor):
    """Actor with Gaussian prediction head."""

    def __init__(self, config: BasicMergeActorConfig) -> None:
        """__init__.

        Args:
            config (BasicMergeActorConfig): config

        Returns:
            None:
        """
        super().__init__()

        # network to go from src, tgt -> embed
        self.src_network = nn.Linear(config.src_size, config.embed_dim)
        self.tgt_network = nn.Linear(config.tgt_size, config.embed_dim)

        # outputs the action after all the compute before it
        self.head = nn.Sequential(
            nn.Linear(2 * config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.act_size * 2),
        )

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
        # shape of src/tgt_embed is [batch_size, obs_size]
        src_embed = torch.mean(
            self.src_network(obs["src"]) * ~(obs["src_mask"][..., None].bool()), dim=-2
        )
        tgt_embed = torch.mean(
            self.tgt_network(obs["tgt"]) * ~(obs["tgt_mask"][..., None].bool()), dim=-2
        )

        # output here is shape [B, act_size * 2]
        output = self.head(torch.cat([src_embed, tgt_embed], dim=-1))

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
