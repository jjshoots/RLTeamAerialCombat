import time
from typing import Literal

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from wingman.replay_buffer import ReplayBuffer
from wingman.utils import cpuize

from dogfighter.models.bases import BaseActor


@torch.no_grad()
def vec_env_collect_to_memory(
    actor: BaseActor,
    vec_env: VectorEnv,
    memory: ReplayBuffer,
    num_transitions: int,
    random_actions: bool,
) -> tuple[ReplayBuffer, dict[Literal["interactions_per_second"], float]]:
    """Runs the actor in the vector environment and collects transitions.

    This collects `num_transitions` transitions using `num_transitions // vev_env.num_envs` steps.
    Note that it stores the items in the replay buffer in the following order:
    - The first `n` items be for observation. (`n=1` if obs = np.ndarray)
    - The next `n` items be for next observation. (`n=1` if obs = np.ndarray)
    - The next 1 item be for action.
    - The next 1 item be for reward.
    - The next 1 item be for termination.

    Args:
        actor (BaseActor): actor
        vec_env (VectorEnv): vec_env
        memory (ReplayBuffer): memory
        num_steps (int): num_steps
        random_actions (bool): random_actions

    Returns:
        tuple[ReplayBuffer, dict[Literal["interactions_per_second"], float]]:
    """
    # to record times
    start_time = time.time()

    # set to eval and zero grad
    actor.eval()
    actor.zero_grad()

    # init the first obs, infos, and reset masks
    obs, info = vec_env.reset()
    non_reset_envs = np.ones(vec_env.num_envs, dtype=bool)

    for _ in range(num_transitions // vec_env.num_envs):
        # compute an action depending on whether we're exploring or not
        if random_actions:
            # sample an action from the env
            act = vec_env.action_space.sample()
        else:
            # get an action from the actor
            act, _ = actor.sample(*actor(obs))

        # step the transition
        next_obs, rew, term, trunc, info = vec_env.step(act)

        # store stuff in mem
        if isinstance(obs, (np.ndarray, torch.Tensor)):
            memory.push(
                [
                    obs[non_reset_envs, ...],  # pyright: ignore[reportArgumentType]
                    act[non_reset_envs, ...],  # pyright: ignore[reportArgumentType]
                    rew[:, None][non_reset_envs, ...],
                    term[:, None][non_reset_envs, ...],
                    next_obs[non_reset_envs, ...],
                ],
                bulk=True,
            )
        elif isinstance(obs, dict):
            raise NotImplementedError("Not implemented yet.")
        else:
            raise NotImplementedError(
                f"No idea how to deal with observation of type {type(obs)}."
            )

        # new observation is the next observation
        # compute exclusion mask of tuples to ignore in the next iteration
        obs = next_obs
        non_reset_envs = ~term & ~trunc

    # print some recordings
    total_time = time.time() - start_time
    interaction_per_second = num_transitions / total_time
    print(f"Collect Stats: {total_time:.2f}s @ {interaction_per_second} t/s.")

    # return the replay buffer and some information
    info: dict[Literal["interactions_per_second"], float] = dict()
    info["interactions_per_second"] = interaction_per_second
    return memory, info


def vec_env_evaluate(
    actor: BaseActor,
    vec_env: VectorEnv,
    num_episodes: int,
) -> dict[Literal["eval_perf", "mean_episode_length"], float]:
    """Performs an evaluation run using the given actor on the vectorized environment.

    Note that `vec_env.num_envs` must cleanly divide num_episodes.

    Args:
        actor (BaseActor): actor
        vec_env (VectorEnv): vec_env
        num_episodes (int): num_episodes

    Returns:
        dict[Literal["eval_perf", "mean_episode_length"], float]:
    """
    assert (
        (num_episodes / vec_env.num_envs) % 1.0 == 0
    ), f"`num_episodes` ({num_episodes}) must be clean multiple of {vec_env.num_envs}."
    # start the evaluation loops
    num_valid_steps = 0
    cumulative_rewards = 0.0

    for _ in range(num_episodes // vec_env.num_envs):
        # reset things
        obs, _ = vec_env.reset()
        done_envs = np.zeros(vec_env.num_envs, dtype=bool)

        # step for one episode
        while not np.all(done_envs):
            # get an action from the actor
            # this is a tensor
            act = actor.infer(*actor(obs))

            # convert the action to cpu
            act = cpuize(act)

            # step the transition
            next_obs, rew, term, trunc, _ = vec_env.step(act)

            # roll observation and record the done envs
            obs = next_obs
            done_envs |= term | trunc

            # record the cumulative rewards and episode length
            cumulative_rewards += np.sum(rew * (1.0 - done_envs))
            num_valid_steps += np.sum(1.0 - done_envs)

    # arrange the results
    info: dict[Literal["eval_perf", "mean_episode_length"], float] = dict()
    info["eval_perf"] = float(cumulative_rewards / num_episodes)
    info["mean_episode_length"] = float(num_valid_steps / num_episodes)
    print(
        "Evaluation Stats: "
        f"{info['eval_perf']} mean eval score @ {info['mean_episode_length']} mean episode length."
    )
    return info
