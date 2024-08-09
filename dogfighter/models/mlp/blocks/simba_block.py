import torch
from torch import nn


class SimbaBlock(nn.Module):
    """SimbaBlock.

    By Hojoon Lee.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
    ) -> None:
        """__init__.

        Args:
            embed_dim (int): embed_dim
            num_blocks (int): num_blocks

        Returns:
            None:
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.ReLU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): a [..., embed_dim] shaped tensor.

        Returns:
            torch.Tensor: same shape as input.
        """
        for block in self.blocks:
            x = x + block(x)

        return x
