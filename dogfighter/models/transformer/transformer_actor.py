from __future__ import annotations

from typing import Literal

import torch
from wingman import NeuralBlocks

from dogfighter.bases.base_actor import Actor, ActorConfig
from dogfighter.models.transformer.transformer_backbone import \
    TransformerBackbone


class TransformerActorConfig(ActorConfig):
    """TransformerActorConfig."""

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

    def instantiate(self) -> "TransformerActor":
        """instantiate.

        Args:

        Returns:
            "TransformerActor":
        """
        return TransformerActor(self)


class TransformerActor(Actor):
    """Actor with Gaussian prediction head."""

    def __init__(self, config: TransformerActorConfig) -> None:
        """__init__.

        Args:
            config (TransformerActorConfig): config

        Returns:
            None:
        """
        super().__init__()

        # the basic backbone
        self.backbone = TransformerBackbone(
            input_dim=config.embed_dim,
            ff_dim=config.ff_dim,
            num_att_heads=config.num_att_heads,
            num_decode_layers=config.num_decode_layers,
            num_encode_layers=config.num_encode_layers,
        )

        # network to go from src -> embed
        _features_description = [config.src_size, config.embed_dim]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.src_input_network = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        _features_description = [config.tgt_size, config.embed_dim]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.tgt_input_network = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # outputs the action after all the compute before it
        _features_description = [config.embed_dim, config.act_size * 2]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.head = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(
        self,
        obs: dict[Literal["src", "tgt", "mask"], torch.Tensor],
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (TransformerObservation): obs

        Returns:
            torch.Tensor:
        """
        # result of this is [B, embed_dim]
        embedding = self.backbone(
            obs=dict(
                src=self.src_input_network(obs["src"]),
                tgt=self.tgt_input_network(obs["tgt"]),
                mask=obs["mask"],
            )
        )

        # output here is shape [B, act_size * 2]
        output = self.head(embedding)

        # split the actions into mean and variance
        # shape is [B, act_size, 2]
        output = output.reshape(*output.shape[:-1], -1, 2)

        # move the mean_var axis to the front
        # output here is shape [2, B, act_size]
        output = torch.moveaxis(output, -1, 0)

        return output
