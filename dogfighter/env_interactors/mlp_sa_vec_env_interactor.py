import time
from typing import Any, Literal

import numpy as np
import torch
from gymnasium import Env
from gymnasium.vector import VectorEnv
from memorial import ReplayBuffer
from memorial.replay_buffers import FlatReplayBuffer
from wingman.utils import cpuize, gpuize

from dogfighter.env_interactors.base import (CollectionFunctionProtocol,
                                             DisplayFunctionProtocol,
                                             EnvInteractorConfig,
                                             EvaluationFunctionProtocol,
                                             SupportedEnvTypes)
from dogfighter.models.base.base_actor import Actor


class MLPSAEnvInteractorConfig(EnvInteractorConfig):
    variant: Literal["mlp_sa"] = "mlp_sa"  # pyright: ignore

    def get_collection_fn(self) -> CollectionFunctionProtocol:
        return mlp_sa_vec_env_collect

    def get_evaluation_fn(self) -> EvaluationFunctionProtocol:
        return mlp_sa_vec_env_evaluate

    def get_display_fn(self) -> DisplayFunctionProtocol:
        return mlp_sa_env_display


@torch.no_grad()
def mlp_sa_vec_env_collect(
    actor: Actor,
    env: SupportedEnvTypes,
    memory: ReplayBuffer,
    num_transitions: int,
    use_random_actions: bool,
) -> tuple[ReplayBuffer, dict[str, Any]]:
    """Runs the actor in the vector environment and collects transitions.

    This collects `num_transitions` transitions using `num_transitions // vev_env.num_envs` steps.
    Note that it stores the items in the replay buffer in the following order:
    - The first `n` items be for observation. (`n=1` if obs = np.ndarray)
    - The next `n` items be for next observation. (`n=1` if obs = np.ndarray)
    - The next 1 item be for action.
    - The next 1 item be for reward.
    - The next 1 item be for termination.

    Args:
        actor (MlpActor): actor
        env (VectorEnv): env
        memory (FlatReplayBuffer): memory
        num_steps (int): num_steps
        use_random_actions (bool): use_random_actions

    Returns:
        tuple[FlatReplayBuffer, dict[Literal["interactions_per_second"], float]]:
    """
    assert isinstance(env, VectorEnv)
    assert isinstance(memory, FlatReplayBuffer)

    # to record times
    start_time = time.time()

    # set to eval and zero grad
    actor.eval()
    actor.zero_grad()

    # list to store memory before big push at end
    transitions = []

    # init the first obs, infos, and reset masks
    obs, _ = env.reset()
    non_reset_envs = np.ones(env.num_envs, dtype=bool)

    for _ in range(num_transitions // env.num_envs):
        # compute an action depending on whether we're exploring or not
        if use_random_actions:
            # sample an action from the env
            act = env.action_space.sample()
        else:
            # get an action from the actor, convert to CPU
            act, _ = actor.sample(*actor(gpuize(obs, actor.device)))
            act = cpuize(act)

        # step the transition
        next_obs, rew, term, trunc, _ = env.step(act)

        # store stuff in mem
        transitions.append(
            (
                obs[non_reset_envs, ...],  # pyright: ignore[reportArgumentType]
                act[non_reset_envs, ...],  # pyright: ignore[reportArgumentType]
                rew[:, None][non_reset_envs, ...],
                term[:, None][non_reset_envs, ...],
                next_obs[non_reset_envs, ...],
            )
        )

        # new observation is the next observation
        # compute exclusion mask of tuples to ignore in the next iteration
        obs = next_obs
        non_reset_envs = ~term & ~trunc

    # push everything to memory in one big go
    memory.push(
        [np.concatenate(items, axis=0) for items in zip(*transitions)],
        bulk=True,
    )

    # record some things
    total_time = time.time() - start_time
    interaction_per_second = num_transitions / total_time

    # return the replay buffer and some information
    return_info = dict()
    return_info["interactions_per_second"] = interaction_per_second
    return memory, return_info


@torch.no_grad()
def mlp_sa_vec_env_evaluate(
    actor: Actor,
    env: SupportedEnvTypes,
    num_episodes: int,
) -> tuple[float, dict[str, Any]]:
    """mlp_sa_vec_env_evaluate.

    Args:
        actor (Actor): actor
        env (SupportedEnvTypes): env
        num_episodes (int): num_episodes

    Returns:
        tuple[float, dict[str, Any]]:
    """
    assert isinstance(env, VectorEnv)
    assert (
        (num_episodes / env.num_envs) % 1.0 == 0
    ), f"`num_episodes` ({num_episodes}) must be clean multiple of {env.num_envs}."

    # set to eval and zero grad
    actor.eval()
    actor.zero_grad()

    # start the evaluation loops
    num_valid_steps = 0
    cumulative_rewards = 0.0

    for _ in range(num_episodes // env.num_envs):
        # reset things
        obs, _ = env.reset()
        done_envs = np.zeros(env.num_envs, dtype=bool)

        # step for one episode
        while not np.all(done_envs):
            # get an action from the actor and convert to CPU
            # this is a tensor
            act = cpuize(actor.infer(*actor(gpuize(obs, actor.device))))

            # step the transition
            next_obs, rew, term, trunc, _ = env.step(act)

            # roll observation and record the done envs
            obs = next_obs
            done_envs |= term | trunc

            # record the cumulative rewards and episode length
            cumulative_rewards += np.sum(rew * (1.0 - done_envs))
            num_valid_steps += np.sum(1.0 - done_envs)

    # arrange the results
    return_info = dict()
    return_info["mean_episode_length"] = float(num_valid_steps / num_episodes)
    mean_cumulative_reward = float(cumulative_rewards / num_episodes)
    return mean_cumulative_reward, return_info


@torch.no_grad()
def mlp_sa_env_display(
    actor: Actor,
    env: SupportedEnvTypes,
) -> None:
    """mlp_sa_env_display.

    Args:
        actor (Actor): actor
        env (SupportedEnvTypes): env

    Returns:
        None:
    """
    assert isinstance(env, Env)

    # set to eval and zero grad
    actor.eval()
    actor.zero_grad()

    # reset things
    term, trunc = False, False
    obs, _ = env.reset()

    # step for one episode
    while not term and not trunc:
        act = cpuize(actor.infer(*actor(gpuize(obs, actor.device))))
        next_obs, _, term, trunc, _ = env.step(act)
        obs = next_obs
