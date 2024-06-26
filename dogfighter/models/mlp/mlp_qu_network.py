import torch
from wingman import NeuralBlocks

from dogfighter.models.bases import Action, BaseCritic
from dogfighter.models.mlp.mlp_bases import MlpEnvParams, MlpModelParams, MlpObservation


class MlpQUNetwork(BaseCritic[MlpObservation]):
    """A classic Q network that uses a transformer backbone."""

    def __init__(
        self,
        env_params: MlpEnvParams,
        model_params: MlpModelParams,
    ) -> None:
        """__init__.

        Args:
            env_params (MlpEnvParams): env_params
            model_params (MlpModelParams): model_params

        Returns:
            None:
        """
        super().__init__(env_params=env_params, model_params=model_params)

        # outputs the action after all the compute before it
        _features_description = [
            env_params.obs_size + env_params.act_size,
            model_params.embed_dim,
            model_params.embed_dim,
            2,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.head = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # register the bias for the uncertainty
        self.register_buffer(
            "uncertainty_bias", torch.tensor(1) * 999.9, persistent=True
        )

    @torch.jit.script
    def forward(
        self,
        obs: MlpObservation,
        act: Action,
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (MlpObservation): obs
            act (Action): Action of shape [B, act_size] or [num_actions, B, act_size]

        Returns:
            torch.Tensor: Q value and Uncertainty tensor of shape [q_u, B] or [q_u, num_actions, B]
        """

        # if we have multiple actions per observation, stack the observation
        if len(act.shape) != len(obs.obs.shape):
            input = obs.obs.expand(act.shape[0], -1, -1)
        else:
            input = obs.obs

        # get the output
        # the shape here is either [B, q_u] or [num_actions, B, q_u]
        q_u = self.head(torch.cat([input, act], dim=-1))

        # move the qu to the first dim
        # the shape here is either [q_u, B] or [q_u, num_actions, B]
        q_u = torch.movedim(q_u, 0, -1)

        # return Q and U
        return q_u
