import torch
import torch.nn as nn

from dogfighter.networks.dataclasses import EnvParams, ModelParams
from dogfighter.networks.qu_network import QUNetwork


class QUEnsemble(nn.Module):
    """Q U Ensemble."""

    def __init__(
        self,
        env_params: EnvParams,
        model_params: ModelParams,
    ) -> None:
        """__init__.

        Args:
            env_params (EnvParams): env_params
            model_params (ModelParams): model_params

        Returns:
            None:
        """
        super().__init__()

        self.networks = nn.ModuleList(
            [
                QUNetwork(
                    env_params=env_params,
                    model_params=model_params,
                )
                for _ in range(model_params.qu_num_ensemble)
            ]
        )

    @torch.jit.script
    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        att: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (torch.Tensor): The observation of the other agents, expected shape is [B, N, embed_dim]
            obs_mask (torch.Tensor): a [B, N, 1] mask tensor for N other UAVs, where True means the observation is null.
            att (torch.Tensor): The attitude of the current agent, expected shape is [B, embed_dim]
            act (torch.Tensor): The action of the current agent, expected shape is [num_actions, B, act_size] or [B, act_size]

        Returns:
            torch.Tensor: Q value and Uncertainty tensor of shape [q_u, B, num_ensemble] or [q_u, num_actions, B, num_ensemble]
        """
        # concatenate the outputs at the last dimension
        # the shape is either [q_u, B, num_ensemble] or [q_u, num_actions, B, num_ensemble]
        output = torch.stack(
            [f(obs=obs, obs_mask=obs_mask, att=att, act=act) for f in self.networks],
            dim=-1,
        )
        return output
