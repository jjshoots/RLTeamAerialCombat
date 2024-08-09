from __future__ import annotations

import torch
from torch import nn


class PreLNDecoder(nn.Module):
    """PreLayerNorm Decoder.

    Inspired from: https://arxiv.org/pdf/2002.04745
    but modified to be pure decoder
    """

    def __init__(
        self,
        dim_model: int,
        dim_feedforward: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        """__init__.

        Args:
            dim_model (int): dim_model
            dim_feedforward (int): dim_feedforward
            num_heads (int): num_heads
            num_layers (int): num_layers

        Returns:
            None:
        """
        super().__init__()
        self._num_layers = num_layers

        # construct the layer norms,
        # we have n `q`s, but only 1 `kv`
        self._q_lns = nn.ModuleList(
            [nn.LayerNorm(dim_model) for _ in range(num_layers)]
        )
        self._k_ln = nn.LayerNorm(dim_model)
        self._v_ln = nn.LayerNorm(dim_model)

        # construct the mha layers
        self._mha_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=dim_model,
                    num_heads=num_heads,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # construct the ffn part
        self._ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim_model),
                    nn.Linear(dim_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, dim_model),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_mask: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            q (torch.Tensor): a [batch_dim, q_len, dim_model] tensor.
            k (torch.Tensor): a [batch_dim, kv_len, dim_model] tensor.
            v (torch.Tensor): a [batch_dim, kv_len, dim_model] tensor.
            k_mask (torch.Tensor): a [batch_dim, kv_len] boolean tensor with False elements indicating unmasked positions.

        Returns:
            torch.Tensor: a [batch_dim, q_len, dim_model] tensor.
        """

        # perform layernorm on kv
        k = self._k_ln(k)
        v = self._v_ln(v)

        # perform decoding
        for q_ln, mha, ffn in zip(self._q_lns, self._mha_layers, self._ffn_layers):
            # residual(prelayernorm + mha)
            q = q + mha(
                query=q_ln(q),
                key=k,
                value=v,
                key_padding_mask=k_mask,
            )[0]

            # residual(prelayernorm + ffn)
            q = q + ffn(q)

        return q
