#!/usr/bin/env python3
import time
import warnings
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from memorial import ReplayBuffer
from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr
from torch.optim.adamw import AdamW
from tqdm import tqdm

from dogfighter.algorithms.base import Algorithm, AlgorithmConfig
from dogfighter.env_interactors.base import UpdateInfos
from dogfighter.models import KnownActorConfigs, KnownCriticConfigs
from dogfighter.models.actors import GaussianActor, GaussianActorConfig
from dogfighter.models.critic_ensemble import CriticEnsemble
from dogfighter.models.critics import UncertaintyAwareCriticConfig
from dogfighter.models.mdp_types import Action, Observation
from dogfighter.models.mlp.mlp_actor import MlpActor


def _mean_numpy_float(x: torch.Tensor) -> float:
    return x.mean().detach().cpu().numpy().item()


class ObsNormalizer(nn.Module):
    def __init__(self, obs_size: int) -> None:
        super().__init__()
        self._count = nn.Parameter(
            torch.tensor(0, dtype=torch.int64),
            requires_grad=False,
        )
        self._mean = nn.Parameter(
            torch.zeros(
                size=(obs_size,),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self._var = nn.Parameter(
            torch.ones(
                size=(obs_size,),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    @property
    def mean(self) -> np.ndarray:
        return self._mean.detach().cpu().numpy()

    @property
    def var(self) -> np.ndarray:
        return self._var.detach().cpu().numpy()

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self._mean) / (self._var + 1e-3).sqrt()

    def update(self, obs: torch.Tensor) -> None:
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        batch_count = obs.size(0)
        new_count = self._count + batch_count

        # compute new mean
        batch_mean = obs.mean(dim=0)
        delta = batch_mean - self._mean
        new_mean = self._mean + delta * batch_count / new_count

        # compute new variance
        batch_var = obs.var(dim=0, unbiased=False)
        new_var = (
            self._var * (self._count / new_count)
            + batch_var * (batch_count / new_count)
            + delta.pow(2) * (self._count / new_count) * (batch_count / new_count)
        )

        # update parameters
        self._mean.copy_(new_mean)
        self._var.copy_(new_var)
        self._count.copy_(new_count)


class CCGEConfig(AlgorithmConfig):
    """Critic Confidence Guided Exploration."""

    variant: Literal["ccge"] = "ccge"  # pyright: ignore
    compile: StrictBool
    device: StrictStr
    actor_config: KnownActorConfigs
    qu_config: KnownCriticConfigs
    qu_num_ensemble: StrictInt
    batch_size: StrictInt
    grad_steps_per_update: StrictInt
    actor_learning_rate: StrictFloat
    critic_learning_rate: StrictFloat
    alpha_learning_rate: StrictFloat
    target_smoothing_coefficient: StrictFloat
    tune_entropy: StrictBool
    target_entropy: StrictFloat
    learn_uncertainty: StrictBool
    discount_factor: StrictFloat
    actor_update_ratio: StrictInt
    critic_update_ratio: StrictInt

    def instantiate(self) -> "CCGE":
        """instantiate.

        Args:

        Returns:
            "CCGE":
        """
        assert isinstance(self.qu_config, UncertaintyAwareCriticConfig)
        assert isinstance(self.actor_config, GaussianActorConfig)
        algorithm = CCGE(self)
        if self.compile:
            torch.compile(algorithm)
        return algorithm


class CCGE(Algorithm):
    """Critic Confidence Guided Exploration."""

    def __init__(self, config: CCGEConfig):
        """__init__.

        Args:
            config (CCGEConfig): config
        """
        super().__init__()
        self.config = config
        self._device = torch.device(config.device)

        # actor head
        self._actor = config.actor_config.instantiate()

        # twin delayed Q networks
        self._critic = CriticEnsemble(
            config.qu_config,
            num_ensemble=config.qu_num_ensemble,
        )
        self._critic_target = CriticEnsemble(
            config.qu_config,
            num_ensemble=config.qu_num_ensemble,
        )

        # HACK: Observation Normalizer
        self._obs_normalizer = ObsNormalizer(obs_size=config.actor_config.obs_size)

        # move the models to the right device
        self._actor.to(self._device)
        self._critic.to(self._device)
        self._critic_target.to(self._device)

        # copy weights and disable gradients for the target network
        self._critic_target.load_state_dict(self._critic.state_dict())
        for param in self._critic_target.parameters():
            param.requires_grad = False

        # tune entropy using log alpha, starts with 0
        if config.target_entropy > 0.0:
            warnings.warn(
                f"Target entropy is recommended to be negative,\
                          currently it is {config.target_entropy},\
                          I hope you know what you're doing..."
            )
        self._log_alpha = nn.Parameter(torch.tensor(0.0, requires_grad=True))

        # define the optimizers
        self._actor_optim = AdamW(
            self._actor.parameters(), lr=config.actor_learning_rate, amsgrad=True
        )
        self._critic_optim = AdamW(
            self._critic.parameters(), lr=config.critic_learning_rate, amsgrad=True
        )
        self._alpha_optim = AdamW(
            [self._log_alpha], lr=config.alpha_learning_rate, amsgrad=True
        )

    @property
    def actor(self) -> GaussianActor:
        """actor.

        Args:

        Returns:
            BaseActor:
        """
        return self._actor

    @property
    def qu_ensemble_critic(self) -> CriticEnsemble:
        """qu_ensemble_critic.

        Args:

        Returns:
            BaseQUEnsemble:
        """
        return self._critic

    def update(
        self,
        memory: ReplayBuffer,
    ) -> UpdateInfos:
        """Updates the model using the replay buffer.

        Note that this expects that `memory` has:
        - The 1st item be for observation.
        - The 2nd item for action.
        - The 3rd item for reward.
        - The 4th item for termination.
        - The 5th item be for next observation.

        Args:
            memory (ReplayBuffer): memory

        Returns:
            Mapping[str, int | float | bool]:
        """
        start_time = time.time()

        # initialise the update infos
        update_info = defaultdict(lambda: 0.0)

        # start the training!
        self.train()
        for _ in tqdm(range(self.config.grad_steps_per_update)):
            obs, act, rew, term, next_obs = memory.sample(self.config.batch_size)
            obs = self._obs_normalizer.normalize(obs)
            next_obs = self._obs_normalizer.normalize(next_obs)

            info = self.forward(
                obs=obs,
                act=act,
                rew=rew,
                term=term,
                next_obs=next_obs,
            )
            for key, value in info.items():
                update_info[key] += value / self.config.grad_steps_per_update

        update_info["steps_per_second"] = self.config.grad_steps_per_update / (
            time.time() - start_time
        )

        return update_info

    def forward(
        self,
        obs: Observation,
        act: Action,
        rew: torch.Tensor,
        term: torch.Tensor,
        next_obs: Observation,
    ) -> dict[str, Any]:
        """The update step disguised as a forward step.

        Args:
            obs (Observation): obs
            act (Action): act
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
        for _ in range(self.config.critic_update_ratio):
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
            all_logs.update(log)

        # update actor
        for _ in range(self.config.actor_update_ratio):
            self._actor_optim.zero_grad()
            loss, log = self._calc_actor_loss(obs=obs, term=term)
            loss.backward()
            self._actor_optim.step()
            all_logs.update(log)

            # also update alpha for entropy
            self._alpha_optim.zero_grad()
            loss, log = self._calc_alpha_loss(obs=obs)
            loss.backward()
            self._alpha_optim.step()
            all_logs.update(log)

        return all_logs

    def _update_q_target(self):
        """update_q_target.

        Args:
            tau:
        """
        # polyak averaging update for target q network
        for target, source in zip(
            self._critic_target.parameters(), self._critic.parameters()
        ):
            target.data.copy_(
                (target.data * (1.0 - self.config.target_smoothing_coefficient))
                + (source.data * self.config.target_smoothing_coefficient)
            )

    def _calc_critic_loss(
        self,
        obs: Observation,
        act: Action,
        rew: torch.Tensor,
        term: torch.Tensor,
        next_obs: Observation,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """_calc_critic_loss.

        Args:
            obs (Observation): obs
            act (Action): act
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
            + (
                -self._log_alpha.exp().detach() * log_probs
                + self.config.discount_factor * next_q
            )
            * term
        )
        q_loss = (current_q - target_q) ** 2

        if self.config.learn_uncertainty:
            # compute the target u and loss
            # target_u is [B, 1]
            # u_loss is [B, num_ensemble]
            target_u = (
                q_loss.mean(dim=-1, keepdim=True).detach()
                + (self.config.discount_factor * next_u * term) ** 2
            ).sqrt()
            u_loss = (current_u - target_u) ** 2
        else:
            target_u = torch.tensor([0.0], device=q_loss.device)
            u_loss = torch.tensor([0.0], device=q_loss.device)

        # sum the losses
        critic_loss = q_loss.mean() + u_loss.mean()

        # some logging parameters
        log = dict()
        log["target_q"] = _mean_numpy_float(target_q)
        log["q_loss"] = _mean_numpy_float(q_loss)
        log["target_u"] = _mean_numpy_float(target_u)
        log["u_loss"] = _mean_numpy_float(u_loss)
        log["critic_loss"] = _mean_numpy_float(critic_loss)

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
        log["actor_loss"] = _mean_numpy_float(actor_loss)

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
        if not self.config.tune_entropy:
            return torch.zeros(1), {}

        # log_probs is shape [B, 1]
        output = self._actor(obs=obs)
        _, log_probs = self._actor.sample(*output)

        # Intuitively, we increase alpha when entropy is less than target entropy, vice versa.
        entropy_loss = -(
            self._log_alpha * (self.config.target_entropy + log_probs).detach()
        ).mean()

        log = dict()
        log["log_alpha"] = self._log_alpha.item()
        log["mean_entropy"] = _mean_numpy_float(-log_probs)
        log["entropy_loss"] = _mean_numpy_float(entropy_loss)

        return entropy_loss, log

    def pick_best_action(
        self,
        obs: Observation,
        act_sets: Action,
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
