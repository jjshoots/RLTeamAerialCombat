import torch
import torch.nn as nn

from dogfighter.models.bases import Action, BaseQUEnsemble
from dogfighter.models.mlp.mlp_bases import (MlpEnvParams, MlpModelParams,
                                             MlpObservation)
from dogfighter.models.mlp.mlp_qu_network import MlpQUNetwork


class MlpQUEnsemble(BaseQUEnsemble[MlpObservation]):
    """Q U Ensemble."""

    def __init__(
        self,
        env_params: MlpEnvParams,
        model_params: MlpModelParams,
    ) -> None:
        super().__init__(env_params=env_params, model_params=model_params)

        self.networks = nn.ModuleList(
            [
                MlpQUNetwork(
                    env_params=env_params,
                    model_params=model_params,
                )
                for _ in range(model_params.qu_num_ensemble)
            ]
        )

    def forward(
        self,
        obs: MlpObservation,
        act: Action,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (MlpObservation): obs
            act (Action): act

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
