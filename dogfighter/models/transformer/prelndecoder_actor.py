from typing import Literal

from pydantic import StrictInt
import torch
from torch import nn

from dogfighter.models.base.base_actor import Actor, ActorConfig
from dogfighter.models.transformer.blocks.pre_ln_decoder import PreLNDecoder


class PreLNDecoderActorConfig(ActorConfig):
    """PreLNDecoderActorConfig."""

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


class PreLNDecoderActor(Actor):
    """A pre-layernorm decoder actor."""

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
        self.src_network = nn.Sequential(
            nn.Linear(config.src_size, config.ff_dim),
            nn.ReLU(),
            nn.Linear(config.ff_dim, config.embed_dim),
        )
        self.tgt_network = nn.Sequential(
            nn.Linear(config.tgt_size, config.ff_dim),
            nn.ReLU(),
            nn.Linear(config.ff_dim, config.embed_dim),
        )

        # outputs the action after all the compute before it
        self.head = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.ReLU(),
            nn.Linear(config.ff_dim, config.act_size * 2),
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
        # generate qkv tensors [B, N, embed_dim] for qkv
        q = self.tgt_network(obs["tgt"])
        kv = self.src_network(obs["src"])

        # pass the tensors into the decoder
        # the result here is [B, N, embed_dim]
        # take the last element over the tgt output
        obs_embed = self.decoder(q=q, k=kv, v=kv, k_mask=obs["src_mask"].bool())
        obs_embed = obs_embed[..., -1, :]

        # output here is shape [B, act_size * 2]
        output = self.head(obs_embed)

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
