import time
import warnings
from collections import defaultdict
from typing import Literal, Sequence, cast

import torch
import torch.nn as nn
from memorial import ReplayBuffer
from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr
from torch.cuda import CUDAGraph
from torch.optim.adamw import AdamW
from tqdm import tqdm

from dogfighter.algorithms.base import Algorithm, AlgorithmConfig
from dogfighter.algorithms.utils import (
    NestedTensor,
    copy_from_memory,
    zeros_from_memory,
)
from dogfighter.env_interactors.base import UpdateInfos
from dogfighter.models import KnownActorConfigs, KnownCriticConfigs
from dogfighter.models.actors import GaussianActor, GaussianActorConfig
from dogfighter.models.critic_ensemble import CriticEnsemble
from dogfighter.models.critics import UncertaintyAwareCriticConfig
from dogfighter.models.mdp_types import Action, Observation


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
        return CCGE(self)


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
        self._actor = config.actor_config.instantiate().to(self._device)

        # twin delayed Q networks
        self._critic = CriticEnsemble(
            config.qu_config,
            num_ensemble=config.qu_num_ensemble,
        ).to(self._device)
        self._critic_target = CriticEnsemble(
            config.qu_config,
            num_ensemble=config.qu_num_ensemble,
        ).to(self._device)

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
        self._log_alpha = nn.Parameter(
            torch.tensor(0.0, requires_grad=True, device=self._device)
        )

        # define the optimizers
        self._actor_optim = AdamW(
            self._actor.parameters(),
            lr=config.actor_learning_rate,
            amsgrad=True,
            capturable=True,
        )
        self._critic_optim = AdamW(
            self._critic.parameters(),
            lr=config.critic_learning_rate,
            amsgrad=True,
            capturable=True,
        )
        self._alpha_optim = AdamW(
            [self._log_alpha],
            lr=config.alpha_learning_rate,
            amsgrad=True,
            capturable=True,
        )

        # for compile
        self.cuda_graph: None | torch.cuda.CUDAGraph = None
        self.infos_ref: None | dict[str, torch.Tensor] = None
        self.batch_ref: None | Sequence[NestedTensor | torch.Tensor] = None
        if config.compile:
            self.compile()

    @property
    def actor(self) -> GaussianActor:
        """actor.

        Args:

        Returns:
            BaseActor:
        """
        return cast(GaussianActor, self._actor)

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
        tensor_update_info = defaultdict(lambda: torch.tensor(0.0, device=self._device))

        if not self.compile:
            # uncompiled train
            self.train()
            for _ in tqdm(range(self.config.grad_steps_per_update)):
                obs, act, rew, term, next_obs = memory.sample(self.config.batch_size)
                info = self.forward(
                    obs=obs,
                    act=act,
                    rew=rew,
                    term=term,
                    next_obs=next_obs,
                )

                # gather infos
                for key, value in info.items():
                    tensor_update_info[key] += value / self.config.grad_steps_per_update

        else:
            # compiled train
            # construct the graph if necessary
            if self.cuda_graph is None:
                # sample a batch, make the batch ref, copy to batch ref
                batch = memory.sample(self.config.batch_size)
                self.batch_ref = [zeros_from_memory(x) for x in batch]
                [copy_from_memory(s, t) for s, t in zip(batch, self.batch_ref)]

                # warmup iteration
                self.forward(*self.batch_ref)  # pyright: ignore[reportArgumentType]

                # construct the graph
                torch.cuda.synchronize()
                self.cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.cuda_graph):
                    self.infos_ref = self.forward(*self.batch_ref)  # pyright: ignore[reportArgumentType]
                torch.cuda.synchronize()

            # cast some things so pyright doesn't scream
            self.cuda_graph = cast(CUDAGraph, self.cuda_graph)
            self.infos_ref = cast(dict[str, torch.Tensor], self.infos_ref)
            self.batch_ref = cast(Sequence[NestedTensor | torch.Tensor], self.batch_ref)

            # start the training!
            torch.cuda.synchronize()
            self.train()
            for _ in tqdm(range(self.config.grad_steps_per_update)):
                batch = memory.sample(self.config.batch_size)
                [copy_from_memory(s, t) for s, t in zip(batch, self.batch_ref)]
                self.cuda_graph.replay()

                # gather infos
                for key, value in self.infos_ref.items():
                    tensor_update_info[key] += value / self.config.grad_steps_per_update

            torch.cuda.synchronize()

        # convert the tensor update info into a float update info
        update_info = dict()
        for key, value in tensor_update_info.items():
            update_info[key] = float(value.detach().cpu().numpy())
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
    ) -> dict[str, torch.Tensor]:
        """The update step disguised as a forward step.

        Args:
            obs (Observation): obs
            act (Action): act
            rew (torch.Tensor): rew
            term (torch.Tensor): term
            next_obs (Observation): next_obs

        Returns:
            dict[str, torch.Tensor]:
        """
        if not self.training:
            raise AssertionError("Model should be in training mode.")

        all_logs = dict()

        # update critic
        for _ in range(self.config.critic_update_ratio):
            self._critic_optim.zero_grad(set_to_none=True)
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
            self._actor_optim.zero_grad(set_to_none=True)
            loss, log = self._calc_actor_loss(obs=obs, term=term)
            loss.backward()
            self._actor_optim.step()
            all_logs.update(log)

            # also update alpha for entropy
            self._alpha_optim.zero_grad(set_to_none=True)
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """_calc_critic_loss.

        Args:
            obs (Observation): obs
            act (Action): act
            rew (torch.Tensor): reward of shape [B, 1]
            term (torch.Tensor): term of shape [B, 1]
            next_obs (Observation): next_obs

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        log: dict[str, torch.Tensor] = dict()
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """_calc_actor_loss.

        Args:
            obs (Observation): obs
            term (torch.Tensor): term

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        log: dict[str, torch.Tensor] = dict()
        log["actor_loss"] = actor_loss.mean().detach()

        return actor_loss, log

    def _calc_alpha_loss(
        self,
        obs: Observation,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """_calc_alpha_loss.

        Args:
            obs (Observation): obs

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        log: dict[str, torch.Tensor] = dict()
        log["log_alpha"] = self._log_alpha.detach()
        log["mean_entropy"] = -log_probs.mean().detach()
        log["entropy_loss"] = entropy_loss.mean().detach()

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
