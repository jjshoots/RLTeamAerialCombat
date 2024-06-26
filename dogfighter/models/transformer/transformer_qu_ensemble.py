import torch
import torch.nn as nn

from dogfighter.models.bases import Action, BaseQUEnsemble
from dogfighter.models.transformer.transformer_bases import (
    TransformerEnvParams, TransformerModelParams, TransformerObservation)
from dogfighter.models.transformer.transformer_qu_network import \
    TransformerQUNetwork


class TransformerQUEnsemble(BaseQUEnsemble[TransformerObservation]):
    """Q U Ensemble."""

    def __init__(
        self,
        env_params: TransformerEnvParams,
        model_params: TransformerModelParams,
    ) -> None:
        """__init__.

        Args:
            env_params (TransformerEnvParams): env_params
            model_params (TransformerEnvParams): model_params

        Returns:
            None:
        """
        super().__init__(env_params=env_params, model_params=model_params)

        self.networks = nn.ModuleList(
            [
                TransformerQUNetwork(
                    env_params=env_params,
                    model_params=model_params,
                )
                for _ in range(model_params.qu_num_ensemble)
            ]
        )

    def forward(
        self,
        obs: TransformerObservation,
        act: Action,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (TransformerObservation): obs
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
