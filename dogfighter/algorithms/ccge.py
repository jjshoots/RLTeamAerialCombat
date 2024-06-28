#!/usr/bin/env python3
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import Field, StrictBool, StrictFloat, StrictInt

from dogfighter.models.bases import (Action, AlgorithmParams, BaseActor,
                                     BaseQUEnsemble, EnvParams, ModelParams,
                                     Observation)


class CCGEParams(AlgorithmParams):
    """CCGEParams."""

    learning_rate: StrictFloat = Field(default=0.003)
    alpha_learning_rate: StrictFloat = Field(default=0.01)
    entropy_tuning: StrictBool = Field(default=True)
    target_entropy: None | StrictFloat = Field(default=None)
    learn_uncertainty: StrictBool = Field(default=False)
    discount_factor: StrictFloat = Field(default=0.99)
    actor_update_ratio: StrictInt = Field(default=1)
    critic_update_ratio: StrictInt = Field(default=1)


class CCGE(nn.Module):
    """Critic Confidence Guided Exploration."""

    def __init__(
        self,
        actor_type: type[BaseActor],
        critic_type: type[BaseQUEnsemble],
        env_params: EnvParams,
        model_params: ModelParams,
        algorithm_params: CCGEParams = CCGEParams(),
        device: torch.device = torch.device("cpu"),
        jit: bool = True,
    ):
        """__init__.

        Args:
            actor_type (type[BaseActor]): actor_type
            critic_type (type[BaseQUEnsemble]): critic_type
            env_params (EnvParams): env_params
            model_params (ModelParams): model_params
            algorithm_params (CCGEParams): algorithm_params
            device (torch.device): device
            jit (bool): jit
        """
        super().__init__()
        self._gamma = algorithm_params.discount_factor
        self._entropy_tuning = algorithm_params.entropy_tuning
        self._target_entropy = algorithm_params.target_entropy
        self._learn_uncertainty = algorithm_params.learn_uncertainty
        self._actor_update_ratio = algorithm_params.actor_update_ratio
        self._critic_update_ratio = algorithm_params.critic_update_ratio

        # actor head
        self._actor = actor_type(env_params=env_params, model_params=model_params)

        # twin delayed Q networks
        self._critic = critic_type(env_params=env_params, model_params=model_params)
        self._critic_target = critic_type(
            env_params=env_params, model_params=model_params
        )

        # move the models to the right device
        self._actor.to(device)
        self._critic.to(device)
        self._critic_target.to(device)

        # copy weights and disable gradients for the target network
        self._critic_target.load_state_dict(self._critic.state_dict())
        for param in self._critic_target.parameters():
            param.requires_grad = False

        # tune entropy using log alpha, starts with 0
        if self._target_entropy is None:
            self._target_entropy = -float(env_params.act_size)
        else:
            if self._target_entropy > 0.0:
                warnings.warn(
                    f"Target entropy is recommended to be negative,\
                              currently it is {self._target_entropy},\
                              I hope you know what you're doing..."
                )
        self._log_alpha = nn.Parameter(torch.tensor(0.0, requires_grad=True))

        # define the optimizers
        self._actor_optim = optim.AdamW(
            self._actor.parameters(), lr=algorithm_params.learning_rate, amsgrad=True
        )
        self._critic_optim = optim.AdamW(
            self._critic.parameters(), lr=algorithm_params.learning_rate, amsgrad=True
        )
        self._alpha_optim = optim.AdamW(
            [self._log_alpha], lr=algorithm_params.alpha_learning_rate, amsgrad=True
        )

        if jit:
            self._actor.compile()
            self._critic.compile()
            self._critic_target.compile()

    @property
    def actor(self) -> BaseActor:
        """actor.

        Args:

        Returns:
            BaseActor:
        """
        return self._actor

    @property
    def qu_ensemble_critic(self) -> BaseQUEnsemble:
        """qu_ensemble_critic.

        Args:

        Returns:
            BaseQUEnsemble:
        """
        return self._critic

    def _update_q_target(self, tau=0.02):
        """update_q_target.

        Args:
            tau:
        """
        # polyak averaging update for target q network
        for target, source in zip(
            self._critic_target.parameters(), self._critic.parameters()
        ):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def _calc_critic_loss(
        self,
        obs: Observation,
        act: Action,
        rew: torch.Tensor,
        next_obs: Observation,
        term: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """_calc_critic_loss.

        Args:
            obs (Observation): obs
            act (Action): act
            rew (torch.Tensor): rew
            next_obs (Observation): next_obs
            term (torch.Tensor): term

        Returns:
            tuple[torch.Tensor, dict[str, Any]]:
        """
        term = 1.0 - term

        # current q and u
        # shape is [B, num_ensemble] for both
        current_q, current_u = self._critic(obs=obs, act=act)

        # compute next q and target_q
        with torch.no_grad():
            # sample the next actions based on the current policy
            # log_probs is of shape [B, 1]
            output = self._actor(obs=obs)
            next_act, log_probs = self._actor.sample(*output)

            # get the next q and u lists and get the value, then...
            # shape is [B, num_ensemble] for both
            next_q, next_u = self._critic_target(obs=next_obs, act=next_act)

            # ...take the min among ensembles
            # shape is [B, 1]
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)
            next_u, _ = torch.min(next_u, dim=-1, keepdim=True)

        # compute target q and loss
        # target_q is [B, 1]
        # q_loss is [B, num_ensemble]
        target_q = (
            rew
            + (-self._log_alpha.exp().detach() * log_probs + self._gamma * next_q)
            * term
        )
        q_loss = (current_q - target_q) ** 2

        if self._learn_uncertainty:
            # compute the target u and loss
            # target_u is [B, 1]
            # u_loss is [B, num_ensemble]
            target_u = (
                q_loss.mean(dim=-1, keepdim=True).detach()
                + (self._gamma * next_u * term) ** 2
            ).sqrt()
            u_loss = (current_u - target_u) ** 2
        else:
            u_loss = torch.tensor([0.0], device=q_loss.device)

        # sum the losses
        critic_loss = q_loss.mean() + u_loss.mean()

        # some logging parameters
        log = dict()
        log["target_q"] = target_q.mean().detach()
        log["q_loss"] = q_loss.mean().detach()
        if self._learn_uncertainty:
            log["target_u"] = target_u.mean().detach()
            log["u_loss"] = u_loss.mean().detach()
        log["critic_loss"] = critic_loss.mean().detach()

        return critic_loss, log

    def _calc_actor_loss(
        self,
        obs: Observation,
        term: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """_calc_actor_loss.

        Args:
            obs (Observation): obs
            term (torch.Tensor): term

        Returns:
            tuple[torch.Tensor, dict[str, Any]]:
        """
        term = 1.0 - term

        # We re-sample actions to calculate expectations of Q.
        # log_probs is of shape [B, 1]
        output = self._actor(obs=obs)
        act, log_probs = self._actor.sample(*output)

        # expected q for actions
        # shape is [B, num_ensemble] for both
        expected_q, expected_u = self._critic(obs=obs, act=act)

        """ REINFORCEMENT LOSS """
        # take minimum q
        # shape is [B, 1]
        expected_q, _ = torch.min(expected_q, dim=-1, keepdim=True)

        # reinforcement target is maximization of Q * done
        rnf_loss = -(expected_q * term).mean()

        """ ENTROPY LOSS"""
        # entropy calculation
        ent_loss = self._log_alpha.exp().detach() * log_probs * term
        ent_loss = ent_loss.mean()

        """ TOTAL LOSS DERIVATION"""
        # sum the losses
        actor_loss = rnf_loss + ent_loss

        log = dict()
        log["actor_loss"] = actor_loss.mean().detach()

        return actor_loss, log

    def _calc_alpha_loss(
        self,
        obs: Observation,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """_calc_alpha_loss.

        Args:
            obs (Observation): obs

        Returns:
            tuple[torch.Tensor, dict[str, Any]]:
        """
        if not self._entropy_tuning:
            return torch.zeros(1), {}

        # log_probs is shape [B, 1]
        output = self._actor(obs=obs)
        _, log_probs = self._actor.sample(*output)

        # Intuitively, we increase alpha when entropy is less than target entropy, vice versa.
        entropy_loss = -(
            self._log_alpha * (self._target_entropy + log_probs).detach()
        ).mean()

        log = dict()
        log["log_alpha"] = self._log_alpha.item()
        log["mean_entropy"] = -log_probs.mean().detach()
        log["entropy_loss"] = entropy_loss.mean().detach()

        return entropy_loss, log

    def update(
        self,
        obs: Observation,
        act: Action,
        next_obs: Observation,
        term: torch.Tensor,
        rew: torch.Tensor,
    ) -> dict[str, Any]:
        """update.

        Args:
            obs (Observation): obs
            act (Action): act
            next_obs (Observation): next_obs
            term (torch.Tensor): term
            rew (torch.Tensor): rew

        Returns:
            dict[str, Any]:
        """
        if not self.training:
            raise AssertionError("Model should be in training mode.")

        all_logs = dict()

        # update critic
        for _ in range(self._critic_update_ratio):
            self._critic_optim.zero_grad()
            loss, log = self._calc_critic_loss(
                obs=obs,
                act=act,
                rew=rew,
                next_obs=next_obs,
                term=term,
            )
            loss.backward()
            self._critic_optim.step()
            self._update_q_target()
            all_logs = {**all_logs, **log}

        # update actor
        for _ in range(self._actor_update_ratio):
            self._actor_optim.zero_grad()
            loss, log = self._calc_actor_loss(obs=obs, term=term)
            loss.backward()
            self._actor_optim.step()
            all_logs = {**all_logs, **log}

            # also update alpha for entropy
            self._alpha_optim.zero_grad()
            loss, log = self._calc_alpha_loss(obs=obs)
            loss.backward()
            self._alpha_optim.step()
            all_logs = {**all_logs, **log}

        return all_logs

    def pick_best_action(
        self,
        obs: Observation,
        act_sets: Action,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """pick_best_action.

        Args:
            obs (Observation): obs
            act_sets (Action): act_sets

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
        """
        # pass things through the critic
        # both here are [num_actions, B, num_ensemble]
        q: torch.Tensor
        u: torch.Tensor
        q, u = self._critic(obs=obs, act=act_sets)

        # total upperbound is just mean Q plus maximum uncertainty
        # shape here is [num_actions, B]
        prospective_q = q.mean(dim=-1, keepdim=False) + u.max(dim=-1, keepdim=False)

        # pick the indices with the highest prospective q
        # shape here is [batch]
        _, indices = torch.max(prospective_q, dim=0, keepdim=False)

        # pick the best actions, shape is [B, act_size]
        best_actions = act_sets[indices, torch.arange(act_sets.shape[1])]
        return best_actions, indices
