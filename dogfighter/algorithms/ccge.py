#!/usr/bin/env python3
import warnings
from dataclasses import field
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from wingman.replay_buffer import ReplayBuffer

from dogfighter.bases.base_actor import Actor, ActorConfig
from dogfighter.bases.base_algorithm import Algorithm, AlgorithmConfig
from dogfighter.bases.base_critic import QUNetworkConfig
from dogfighter.bases.base_types import Observation
from dogfighter.models.qu_ensemble import QUEnsemble


class CCGEConfig(AlgorithmConfig):
    """Critic Confidence Guided Exploration."""

    device: str
    actor_config: ActorConfig
    qu_config: QUNetworkConfig
    qu_num_ensemble: int = field(default=2)
    learning_rate: float = field(default=0.003)
    alpha_learning_rate: float = field(default=0.01)
    tune_entropy: bool = field(default=True)
    target_entropy: float = field(default=0.0)
    learn_uncertainty: bool = field(default=True)
    discount_factor: float = field(default=0.99)
    actor_update_ratio: int = field(default=1)
    critic_update_ratio: int = field(default=1)

    def instantiate(self) -> "CCGE":
        """instantiate.

        Args:

        Returns:
            "CCGE":
        """
        return CCGE(
            device=torch.device(self.device),
            actor=self.actor_config.instantiate(),
            ensemble_qu=QUEnsemble(self.qu_config, num_ensemble=self.qu_num_ensemble),
            ensemble_qu_target=QUEnsemble(
                self.qu_config, num_ensemble=self.qu_num_ensemble
            ),
            learning_rate=self.learning_rate,
            alpha_learning_rate=self.alpha_learning_rate,
            tune_entropy=self.tune_entropy,
            target_entropy=self.target_entropy,
            learn_uncertainty=self.learn_uncertainty,
            discount_factor=self.discount_factor,
            actor_update_ratio=self.actor_update_ratio,
            critic_update_ratio=self.critic_update_ratio,
        )


class CCGE(Algorithm):
    """Critic Confidence Guided Exploration."""

    def __init__(
        self,
        device: torch.device,
        actor: Actor,
        ensemble_qu: QUEnsemble,
        ensemble_qu_target: QUEnsemble,
        learning_rate: float,
        alpha_learning_rate: float,
        tune_entropy: bool,
        target_entropy: float,
        learn_uncertainty: bool,
        discount_factor: float,
        actor_update_ratio: int,
        critic_update_ratio: int,
    ):
        """__init__.

        Args:
            actor (Actor): actor
            ensemble_qu (QUEnsemble): ensemble_qu
            ensemble_qu_target (QUEnsemble): ensemble_qu_target
            learning_rate (float): learning_rate
            alpha_learning_rate (float): alpha_learning_rate
            tune_entropy (bool): tune_entropy
            target_entropy (float): target_entropy
            learn_uncertainty (bool): learn_uncertainty
            discount_factor (float): discount_factor
            actor_update_ratio (int): actor_update_ratio
            critic_update_ratio (int): critic_update_ratio
            device (torch.device): device
        """
        super().__init__()
        self._gamma = discount_factor
        self._tune_entropy = tune_entropy
        self._target_entropy = target_entropy
        self._learn_uncertainty = learn_uncertainty
        self._actor_update_ratio = actor_update_ratio
        self._critic_update_ratio = critic_update_ratio

        # actor head
        self._actor = actor

        # twin delayed Q networks
        self._critic = ensemble_qu
        self._critic_target = ensemble_qu_target

        # move the models to the right device
        self._actor.to(device)
        self._critic.to(device)
        self._critic_target.to(device)

        # copy weights and disable gradients for the target network
        self._critic_target.load_state_dict(self._critic.state_dict())
        for param in self._critic_target.parameters():
            param.requires_grad = False

        # tune entropy using log alpha, starts with 0
        if self._target_entropy > 0.0:
            warnings.warn(
                f"Target entropy is recommended to be negative,\
                          currently it is {self._target_entropy},\
                          I hope you know what you're doing..."
            )
        self._log_alpha = nn.Parameter(torch.tensor(0.0, requires_grad=True))

        # define the optimizers
        self._actor_optim = optim.AdamW(
            self._actor.parameters(), lr=learning_rate, amsgrad=True
        )
        self._critic_optim = optim.AdamW(
            self._critic.parameters(), lr=learning_rate, amsgrad=True
        )
        self._alpha_optim = optim.AdamW(
            [self._log_alpha], lr=alpha_learning_rate, amsgrad=True
        )

    @property
    def actor(self) -> Actor:
        """actor.

        Args:

        Returns:
            BaseActor:
        """
        return self._actor

    @property
    def qu_ensemble_critic(self) -> QUEnsemble:
        """qu_ensemble_critic.

        Args:

        Returns:
            BaseQUEnsemble:
        """
        return self._critic

    def update(
        self,
        memory: ReplayBuffer,
        batch_size: int,
        num_gradient_steps: int,
    ) -> dict[str, Any]:
        """Updates the model using the replay buffer for `num_gradient_steps`.

        Note that this expects that `memory` has:
        - The first `n` items be for observation.
        - The next `n` items be for next observation.
        - The next 1 item be for action.
        - The next 1 item be for reward.
        - The next 1 item be for termination.

        Args:
            memory (ReplayBuffer): memory
            batch_size (int): batch_size
            num_gradient_steps (int): num_gradient_steps

        Returns:
            dict[str, Any]:
        """
        # initialise the update infos
        update_info = {}

        # start the training!
        self.train()
        for _ in tqdm(range(num_gradient_steps)):
            obs, act, rew, term, next_obs = memory.sample(batch_size)
            update_info = self.forward(
                obs=obs,
                act=act,
                rew=rew,
                term=term,
                next_obs=next_obs,
            )

        return update_info

    def forward(
        self,
        obs: Observation,
        act: torch.Tensor,
        rew: torch.Tensor,
        term: torch.Tensor,
        next_obs: Observation,
    ) -> dict[str, Any]:
        """The update step disguised as a forward step.

        Args:
            obs (Observation): obs
            act (torch.Tensor): act
            rew (torch.Tensor): rew
            term (torch.Tensor): term
            next_obs (Observation): next_obs

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
                term=term,
                next_obs=next_obs,
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
        act: torch.Tensor,
        rew: torch.Tensor,
        term: torch.Tensor,
        next_obs: Observation,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """_calc_critic_loss.

        Args:
            obs (Observation): obs
            act (torch.Tensor): act
            rew (torch.Tensor): reward of shape [B, 1]
            term (torch.Tensor): term of shape [B, 1]
            next_obs (Observation): next_obs

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
            target_u = torch.tensor([0.0], device=q_loss.device)
            u_loss = torch.tensor([0.0], device=q_loss.device)

        # sum the losses
        critic_loss = q_loss.mean() + u_loss.mean()

        # some logging parameters
        log = dict()
        log["target_q"] = target_q.mean().detach()
        log["q_loss"] = q_loss.mean().detach()
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
        if not self._tune_entropy:
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

    def pick_best_action(
        self,
        obs: Observation,
        act_sets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """pick_best_action.

        Args:
            obs (Observation): obs
            act_sets (torch.Tensor): act_sets

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
