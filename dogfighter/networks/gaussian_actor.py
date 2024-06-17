import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func
from wingman import NeuralBlocks

from dogfighter.networks.backbone import Backbone
from dogfighter.networks.dataclasses import EnvParams, ModelParams


class GaussianActor(nn.Module):
    """Actor with Gaussian prediction head."""

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
        # the basic backbone
        self.backbone = Backbone(
            env_params=env_params,
            model_params=model_params,
        )

        # outputs the action after all the compute before it
        _features_description = [model_params.embed_dim, env_params.act_size * 2]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.head = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    @torch.jit.script
    def forward(
        self, obs: torch.Tensor, obs_mask: torch.Tensor, att: torch.Tensor
    ) -> torch.Tensor:
        """forward.

        Args:
            obs (torch.Tensor): Observation of shape [B, N, obs_shape]
            obs_mask (torch.Tensor): Observation mask of shape [B, N, 1]
            att (torch.Tensor): Attitude of shape [B, att_size]

        Returns:
            torch.Tensor: Action of shape [mean_var, B, act_size]
        """
        # embedding here is shape [B, embed_dim]
        embedding = self.backbone(obs=obs, obs_mask=obs_mask, att=att)

        # actions here is shape [B, act_size]
        output = self.head(embedding)

        # split the actions into mean and variance
        # output here is shape [2, B, act_size]
        output = output.reshape(*output.shape[:-1], -1, 2)
        output = torch.moveaxis(output, 0, -1)

        if len(output.shape) > 2:
            output = output.moveaxis(-2, 0)

        return output

    @torch.jit.script
    @staticmethod
    def sample(
        mean: torch.Tensor, var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """sample.

        Args:
            mean (torch.Tensor): mean
            var (torch.Tensor): var

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
        """
        # lower bound sigma and bias it
        normals = dist.Normal(mean, func.softplus(var) + 1e-6)

        # sample from dist
        mu_samples = normals.rsample()
        actions = torch.tanh(mu_samples)

        # calculate log_probs
        log_probs = normals.log_prob(mu_samples) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return actions, log_probs

    @torch.jit.script
    @staticmethod
    def infer(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """infer.

        Args:
            mean (torch.Tensor): mean
            var (torch.Tensor): var

        Returns:
            torch.Tensor:
        """
        return torch.tanh(mean)
