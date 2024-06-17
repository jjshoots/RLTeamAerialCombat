#!/usr/bin/env python3
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from dogfighter.networks.dataclasses import (EnvParams, LearningParams,
                                             ModelParams)
from dogfighter.networks.gaussian_actor import GaussianActor
from dogfighter.networks.qu_ensemble import QUEnsemble


class CCGE(nn.Module):
    """Critic Confidence Guided Exploration."""

    def __init__(
        self,
        env_params: EnvParams,
        model_params: ModelParams,
        learning_params: LearningParams,
    ):
        """__init__.

        Args:
            env_params (EnvParams): env_params
            model_params (ModelParams): model_params
            learning_params (LearningParams): learning_params
        """
        super().__init__()
        self.gamma = learning_params.discount_factor
        self.target_entropy = learning_params.target_entropy
        self.actor_update_ratio = learning_params.actor_update_ratio
        self.critic_update_ratio = learning_params.critic_update_ratio

        # actor head
        self.actor = GaussianActor(env_params=env_params, model_params=model_params)

        # twin delayed Q networks
        self.critic = QUEnsemble(env_params=env_params, model_params=model_params)
        self.critic_target = QUEnsemble(
            env_params=env_params, model_params=model_params
        )

        # copy weights and disable gradients for the target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # tune entropy using log alpha, starts with 0
        if self.target_entropy is None:
            self.target_entropy = -float(env_params.act_size)
        else:
            if self.target_entropy > 0.0:
                warnings.warn(
                    f"Target entropy is recommended to be negative,\
                              currently it is {self.target_entropy},\
                              I hope you know what you're doing..."
                )
        self.log_alpha = nn.Parameter(torch.tensor(0.0, requires_grad=True))

        # define the optimizers
        self.actor_optim = optim.AdamW(
            self.actor.parameters(), lr=learning_params.learning_rate, amsgrad=True
        )
        self.alpha_optim = optim.AdamW(
            self.critic.parameters(), lr=learning_params.learning_rate, amsgrad=True
        )
        self.critic_optim = optim.AdamW(
            [self.log_alpha], lr=learning_params.alpha_learning_rate, amsgrad=True
        )

    def _update_q_target(self, tau=0.02):
        """update_q_target.

        Args:
            tau:
        """
        # polyak averaging update for target q network
        for target, source in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    @torch.jit.script
    def _calc_critic_loss(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        att: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        next_obs_mask: torch.Tensor,
        next_att: torch.Tensor,
        terms: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """calc_critic_loss.

        Args:
            obs (torch.Tensor): Observation of shape [B, N, obs_shape]
            obs_mask (torch.Tensor): Observation mask of shape [B, N, 1]
            att (torch.Tensor): Attitude of shape [B, att_size]
            act (torch.Tensor): Action of shape [B, act_size]
            rew (torch.Tensor): Reward of shape [B, 1]
            next_obs (torch.Tensor):
            next_obs_mask (torch.Tensor):
            next_att (torch.Tensor):
            terms (torch.Tensor): Terminations of shape [B, 1]

        Returns:
            tuple[torch.Tensor, dict]:
        """
        terms = 1.0 - terms

        # current q and u
        # shape is [B, num_ensemble] for both
        current_q, current_u = self.critic(obs=obs, obs_mask=obs_mask, att=att, act=act)

        # compute next q and target_q
        with torch.no_grad():
            # sample the next actions based on the current policy
            # next_actions is of shape [B, act_size]
            # log_probs is of shape [B, 1]
            output = self.actor(obs=obs, obs_mask=obs_mask, att=att)
            next_actions, log_probs = self.actor.sample(*output)

            # get the next q and u lists and get the value, then...
            # shape is [B, num_ensemble] for both
            next_q, next_u = self.critic_target(
                obs=next_obs, obs_mask=next_obs_mask, att=next_att, act=next_actions
            )

            # ...take the min among ensembles
            # shape is [B, 1]
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)
            next_u, _ = torch.min(next_u, dim=-1, keepdim=True)

        # compute target q and loss
        # target_q is [B, 1]
        # q_loss is [B, num_ensemble]
        target_q = (
            rew
            + (-self.log_alpha.exp().detach() * log_probs + self.gamma * next_q) * terms
        )
        q_loss = (current_q - target_q) ** 2

        # compute the target u and loss
        # target_u is [B, 1]
        # u_loss is [B, num_ensemble]
        target_u = (
            q_loss.mean(dim=-1, keepdim=True).detach()
            + (self.gamma * next_u * terms) ** 2
        ).sqrt()
        u_loss = (current_u - target_u) ** 2

        # sum the losses
        critic_loss = q_loss.mean() + u_loss.mean()

        # some logging parameters
        log = dict()
        log["target_q"] = target_q.mean().detach()
        log["target_u"] = target_u.mean().detach()
        log["q_loss"] = q_loss.mean().detach()
        log["u_loss"] = u_loss.mean().detach()
        log["critic_loss"] = critic_loss.mean().detach()

        return critic_loss, log

    @torch.jit.script
    def _calc_actor_loss(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        att: torch.Tensor,
        terms: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """_calc_actor_loss

        Args:
            obs (torch.Tensor): Observation of shape [B, N, obs_shape]
            obs_mask (torch.Tensor): Observation mask of shape [B, N, 1]
            att (torch.Tensor): Attitude of shape [B, att_size]
            terms (torch.Tensor): Terminations of shape [B, 1]

        Returns:
            tuple[torch.Tensor, dict]:
        """
        terms = 1.0 - terms

        # We re-sample actions to calculate expectations of Q.
        # actions is of shape [B, act_size]
        # log_probs is of shape [B, 1]
        output = self.actor(obs=obs, obs_mask=obs_mask, att=att)
        actions, log_probs = self.actor.sample(*output)

        # expected q for actions
        # shape is [B, num_ensemble] for both
        expected_q, expected_u = self.critic(
            obs=obs, obs_mask=obs_mask, att=att, act=actions
        )

        """ REINFORCEMENT LOSS """
        # take minimum q
        # shape is [B, 1]
        expected_q, _ = torch.min(expected_q, dim=-1, keepdim=True)

        # reinforcement target is maximization of Q * done
        rnf_loss = -(expected_q * terms).mean()

        """ ENTROPY LOSS"""
        # entropy calculation
        ent_loss = self.log_alpha.exp().detach() * log_probs * terms
        ent_loss = ent_loss.mean()

        """ TOTAL LOSS DERIVATION"""
        # sum the losses
        actor_loss = rnf_loss + ent_loss

        log = dict()
        log["actor_loss"] = actor_loss.mean().detach()

        return actor_loss, log

    @torch.jit.script
    def _calc_alpha_loss(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        att: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """_calc_alpha_loss

        Args:
            obs (torch.Tensor): Observation of shape [B, N, obs_shape]
            obs_mask (torch.Tensor): Observation mask of shape [B, N, 1]
            att (torch.Tensor): Attitude of shape [B, att_size]

        Returns:
            tuple[torch.Tensor, dict]:
        """
        if not self.entropy_tuning:
            return torch.zeros(1), {}

        # log_probs is shape [B, 1]
        output = self.actor(obs=obs, obs_mask=obs_mask, att=att)
        _, log_probs = self.actor.sample(*output)

        # Intuitively, we increase alpha when entropy is less than target entropy, vice versa.
        entropy_loss = -(
            self.log_alpha * (self.target_entropy + log_probs).detach()
        ).mean()

        log = dict()
        log["log_alpha"] = self.log_alpha.item()
        log["mean_entropy"] = -log_probs.mean().detach()
        log["entropy_loss"] = entropy_loss.mean().detach()

        return entropy_loss, log

    @torch.jit.script
    def update(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        att: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        next_obs_mask: torch.Tensor,
        next_att: torch.Tensor,
        terms: torch.Tensor,
    ) -> dict[str, Any]:
        """Update the entire model.

        Args:
            obs (torch.Tensor): obs
            obs_mask (torch.Tensor): obs_mask
            att (torch.Tensor): att
            act (torch.Tensor): act
            rew (torch.Tensor): rew
            next_obs (torch.Tensor): next_obs
            next_obs_mask (torch.Tensor): next_obs_mask
            next_att (torch.Tensor): next_att
            terms (torch.Tensor): terms

        Returns:
            dict[str, Any]:
        """
        if not self.training:
            raise AssertionError("Model should be in training mode.")

        all_logs = dict()

        # update critic
        for _ in range(self.critic_update_ratio):
            self.critic_optim.zero_grad()
            loss, log = self._calc_critic_loss(
                obs=obs,
                obs_mask=obs_mask,
                att=att,
                act=act,
                rew=rew,
                next_obs=next_obs,
                next_obs_mask=next_obs_mask,
                next_att=next_att,
                terms=terms,
            )
            loss.backward()
            self.critic_optim.step()
            self._update_q_target()
            all_logs = {**all_logs, **log}

        # update actor
        for _ in range(self.actor_update_ratio):
            self.actor_optim.zero_grad()
            loss, log = self._calc_actor_loss(
                obs=obs,
                obs_mask=obs_mask,
                att=att,
                terms=terms,
            )
            loss.backward()
            self.actor_optim.step()
            all_logs = {**all_logs, **log}

            # also update alpha for entropy
            self.alpha_optim.zero_grad()
            loss, log = self._calc_alpha_loss(
                obs=obs,
                obs_mask=obs_mask,
                att=att,
            )
            loss.backward()
            self.alpha_optim.step()
            all_logs = {**all_logs, **log}

        return all_logs

    def pick_best_action(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        att: torch.Tensor,
        act_sets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """pick_best_action.

        Args:
            obs (torch.Tensor): Observation of shape [B, N, obs_shape]
            obs_mask (torch.Tensor): Observation mask of shape [B, N, 1]
            att (torch.Tensor): Attitude of shape [B, att_size]
            act_sets (torch.Tensor): Set of actions of shape [num_actions, B, act_size]

        Returns:
            tuple[torch.FloatTensor, torch.FloatTensor]:
            - [B, act_size] array for best actions
            - [B, ] integer array specifying which action in the batch dimension came from which action set index
        """
        # pass things through the critic
        # both here are [num_actions, B, num_ensemble]
        q: torch.Tensor
        u: torch.Tensor
        q, u = self.critic(obs=obs, obs_mask=obs_mask, att=att, act=act_sets)

        # total upperbound is just mean Q plus maximum uncertainty
        # shape here is [num_actions, B]
        prospective_q = q.mean(dim=-1, keepdim=False) + u.max(dim=-1, keepdim=False)

        # pick the indices with the highest prospective q
        # shape here is [batch]
        _, indices = torch.max(prospective_q, dim=0, keepdim=False)

        # pick the best actions, shape is [B, act_size]
        best_actions = act_sets[indices, torch.arange(obs.shape[0])]
        return best_actions, indices
