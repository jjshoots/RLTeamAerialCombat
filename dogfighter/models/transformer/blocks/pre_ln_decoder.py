import torch
from torch import nn


class PreLNDecoder(nn.Module):
    """PreLayerNorm Decoder.

    Inspired from: https://arxiv.org/pdf/2002.04745
    but modified to be pure decoder
    """

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        num_heads: int,
    ) -> None:
        """__init__.

        Args:
            embed_dim (int): embed_dim
            ff_dim (int): ff_dim
            num_heads (int): num_heads

        Returns:
            None:
        """
        super().__init__()

        # construct the layernorms
        self._q_ln = nn.LayerNorm(embed_dim)
        self._k_ln = nn.LayerNorm(embed_dim)
        self._v_ln = nn.LayerNorm(embed_dim)

        # construct the mha layers
        self._mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # output layernorm and ff
        self._ff_ln = nn.LayerNorm(embed_dim)
        self._ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
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
            q (torch.Tensor): a [batch_dim, q_len, embed_dim] tensor.
            k (torch.Tensor): a [batch_dim, kv_len, embed_dim] tensor.
            v (torch.Tensor): a [batch_dim, kv_len, embed_dim] tensor.
            k_mask (torch.Tensor): a [batch_dim, kv_len] boolean tensor with False elements indicating unmasked positions.

        Returns:
            torch.Tensor: a [batch_dim, q_len, embed_dim] tensor.
        """
        # perform layernorm then decode
        q = (
            q
            + self._mha(
                query=self._q_ln(q),
                key=self._k_ln(k),
                value=self._v_ln(v),
                key_padding_mask=k_mask,
            )[0]
        )

        # feedforward layers
        q = q + self._ff(self._ff_ln(q))

        return q
