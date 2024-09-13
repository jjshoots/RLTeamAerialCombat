from typing import Generic

import torch
import torch.nn as nn

from dogfighter.algorithms.base import Action, Observation
from dogfighter.models.base.base import CriticConfig


class CriticEnsemble(nn.Module, Generic[Observation, Action]):
    """CriticEnsemble."""

    def __init__(
        self,
        base_critic_config: CriticConfig,
        num_ensemble: int,
    ) -> None:
        """__init__.

        Args:
            base_critic_config (CriticConfig): base_network_config
            num_ensemble (int): num_ensemble

        Returns:
            None:
        """
        super().__init__()

        self.networks = nn.ModuleList(
            [base_critic_config.instantiate() for _ in range(num_ensemble)]
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
            torch.Tensor: Tensor of shape [q_u, B, num_ensemble] or [q_u, num_actions, B, num_ensemble]
        """
        # concatenate the outputs at the last dimension
        # the shape is either [..., B, num_ensemble] or [..., num_actions, B, num_ensemble]
        output = torch.stack(
            [f(obs=obs, act=act) for f in self.networks],
            dim=-1,
        )
        return output
