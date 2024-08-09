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
        # we have n `v`s, but only 1 `q` and `k`
        self._q_ln = nn.LayerNorm(dim_model)
        self._k_ln = nn.LayerNorm(dim_model)
        self._v_lns = nn.ModuleList(
            [nn.LayerNorm(dim_model) for _ in range(num_layers)]
        )

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
            q (torch.Tensor): a [batch_dim, qk_len, dim_model] tensor.
            k (torch.Tensor): a [batch_dim, qk_len, dim_model] tensor.
            v (torch.Tensor): a [batch_dim, v_len, dim_model] tensor.
            k_mask (torch.Tensor): a [batch_dim, qk_len] boolean tensor with False elements indicating unmasked positions.

        Returns:
            torch.Tensor: a [batch_dim, v_len, dim_model] tensor.
        """

        # perform layernorm on the q and k
        q = self._q_ln(q)
        k = self._k_ln(k)

        # perform decoding
        for v_ln, mha, ffn in zip(self._v_lns, self._mha_layers, self._ffn_layers):
            # residual(prelayernorm + mha)
            v = v + mha(
                query=q,
                key=k,
                value=v_ln(v),
                key_padding_mask=k_mask,
            )[0]

            # residual(prelayernorm + ffn)
            v = v + ffn(v)

        return v
