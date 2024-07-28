from __future__ import annotations

from typing import Literal

import torch
from wingman import NeuralBlocks

from dogfighter.bases.base_critic import QUNetwork, QUNetworkConfig
from dogfighter.models.transformer.transformer_backbone import \
    TransformerBackbone


class TransformerQUNetworkConfig(QUNetworkConfig):
    """TransformerQUNetworkConfig."""

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

    def instantiate(self) -> "TransformerQUNetwork":
        """instantiate.

        Args:

        Returns:
            "TransformerQUNetwork":
        """
        return TransformerQUNetwork(
            src_size=self.src_size,
            tgt_size=self.tgt_size,
            act_size=self.act_size,
            embed_dim=self.embed_dim,
            backbone=TransformerBackbone(
                input_dim=self.embed_dim,
                ff_dim=self.ff_dim,
                num_att_heads=self.num_att_heads,
                num_decode_layers=self.num_decode_layers,
                num_encode_layers=self.num_encode_layers,
            ),
        )


class TransformerQUNetwork(QUNetwork):
    """A classic Q network that uses a transformer backbone."""

    def __init__(
        self,
        src_size: int,
        tgt_size: int,
        act_size: int,
        embed_dim: int,
        backbone: TransformerBackbone,
    ) -> None:
        """__init__.

        Args:
            src_size (int): src_size
            tgt_size (int): tgt_size
            act_size (int): act_size
            embed_dim (int): embed_dim
            backbone (TransformerBackbone): backbone

        Returns:
            None:
        """
        super().__init__()

        # the basic backbone
        self.backbone = backbone

        # network to go from src -> embed
        _features_description = [src_size, embed_dim]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.src_input_network = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        _features_description = [tgt_size, embed_dim]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.tgt_input_network = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # network to get the action representation
        _features_description = [act_size, embed_dim]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.act_network = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # network to merge the action and obs/att representations
        _features_description = [2 * embed_dim, 2]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.embed_act_merger = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # register the bias for the uncertainty
        self.register_buffer(
            "uncertainty_bias", torch.tensor(1) * 999.9, persistent=True
        )

    def forward(
        self,
        obs: dict[Literal["src", "tgt", "mask"], torch.Tensor],
        act: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (dict[Literal["src", "tgt", "mask"], torch.Tensor]): [B, N, obs_size] tensors
            act (torch.Tensor): Action of shape [B, act_size] or [num_actions, B, act_size]

        Returns:
            torch.Tensor: Q value and Uncertainty tensor of shape [q_u, B] or [q_u, num_actions, B]
        """
        # TODO: fix this

        # pass obs and att through the backbone
        # the shape here is [B, embed_dim]
        obs_embed = self.backbone(
            obs=dict(
                src=self.src_input_network(obs["src"]),
                tgt=self.tgt_input_network(obs["tgt"]),
                mask=obs["mask"],
            )
        )

        # pass the action through the action network
        # the shape here is either [B, embed_dim] or [num_actions, B, embed_dim]
        act_embed = self.act_network(act)

        # if we have multiple actions per observation, stack the observation
        if len(act.shape) != len(obs["src"].shape):
            obs_embed = obs_embed.expand(act.shape[0], -1, -1)

        # merge things together and get the output
        # the shape here is either [B, q_u] or [num_actions, B, q_u]
        q_u = self.head(torch.cat([obs_embed, act_embed], dim=-1))

        # move the qu to the first dim
        # the shape here is either [q_u, B] or [q_u, num_actions, B]
        q_u = torch.movedim(q_u, 0, -1)

        # return Q and U
        return q_u
