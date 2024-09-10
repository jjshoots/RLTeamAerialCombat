from typing import Generic, TypeVar

import torch
import torch.nn as nn

from dogfighter.models.base.base_critic import QUNetworkConfig

Observation = TypeVar("Observation")
Action = TypeVar("Action")


class QUEnsemble(nn.Module, Generic[Observation, Action]):
    """Q U Ensemble."""

    def __init__(
        self,
        base_network_config: QUNetworkConfig,
        num_ensemble: int,
    ) -> None:
        """__init__.

        Args:
            base_network_config (QUNetworkConfig): base_network_config
            num_ensemble (int): num_ensemble

        Returns:
            None:
        """
        super().__init__()

        self.networks = nn.ModuleList(
            [base_network_config.instantiate() for _ in range(num_ensemble)]
        )

    def forward(
        self,
        obs: Observation,
        act: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (Observation): obs
            act (torch.Tensor): act

        Returns:
            torch.Tensor: Q value and Uncertainty tensor of shape [q_u, B, num_ensemble] or [q_u, num_actions, B, num_ensemble]
        """
        # concatenate the outputs at the last dimension
        # the shape is either [q_u, B, num_ensemble] or [q_u, num_actions, B, num_ensemble]
        output = torch.stack(
            [f(obs=obs, act=act) for f in self.networks],
            dim=-1,
        )
        return output
