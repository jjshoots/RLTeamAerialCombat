from typing import OrderedDict

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from wingman import ReplayBuffer
from wingman.utils import cpuize

from dogfighter.models.bases import BaseActor


@torch.no_grad()
def env_collect_to_memory(
    actor: BaseActor,
    vec_env: VectorEnv,
    device: torch.device,
    memory: ReplayBuffer,
    random_actions: bool,
    num_steps: int,
) -> ReplayBuffer:
    """Runs the actor in the vector environment and collects transitions.

    This collects `num_steps * vec_env.num_envs` transitions.
    Note that it stores the items in the replay buffer in the following order:
    - The first `n` items be for observation. (`n=1` if obs = np.ndarray)
    - The next `n` items be for next observation. (`n=1` if obs = np.ndarray)
    - The next 1 item be for action.
    - The next 1 item be for reward.
    - The next 1 item be for termination.

    Args:
        actor (BaseActor): actor
        vec_env (VectorEnv): vec_env
        device (torch.device): device
        memory (ReplayBuffer): memory
        random_actions (bool): random_actions
        num_steps (int): num_steps

    Returns:
        ReplayBuffer:
    """
    # set to eval and zero grad
    actor.eval()
    actor.zero_grad()

    # init the first obs, infos, and reset masks
    obs, info = vec_env.reset()
    non_reset_envs = np.ones(vec_env.num_envs, dtype=bool)

    for i in range(num_steps):
        # compute an action depending on whether we're exploring or not
        if random_actions:
            # sample an action from the env
            act = vec_env.action_space.sample()
        else:
            # get an action from the actor
            # this is a tensor
            policy_observation = actor.package_observation(obs, device)
            act, _ = actor.sample(*actor(policy_observation))

            # convert the action to cpu
            act = cpuize(act)

        # step the transition
        next_obs, rew, term, trunc, info = vec_env.step(act)

        # store stuff in mem
        if isinstance(obs, np.ndarray):
            memory.push(
                [
                    obs[non_reset_envs, ...],
                    next_obs[non_reset_envs, ...],
                    act[non_reset_envs, ...],
                    np.expand_dims(rew, axis=-1)[non_reset_envs, ...],
                    np.expand_dims(term, axis=-1)[non_reset_envs, ...],
                ],
                bulk=True,
            )
        elif isinstance(obs, OrderedDict):
            memory.push(
                [
                    *obs[non_reset_envs, ...].values(),
                    *next_obs[non_reset_envs, ...].values(),
                    act[non_reset_envs, ...],
                    np.expand_dims(rew, axis=-1)[non_reset_envs, ...],
                    np.expand_dims(term, axis=-1)[non_reset_envs, ...],
                ],
                bulk=True,
            )
        else:
            raise NotImplementedError(
                f"No idea how to deal with observation of type {type(obs)}."
            )

        # new observation is the next observation
        # compute exclusion mask of tuples to ignore in the next iteration
        obs = next_obs
        non_reset_envs = ~term & ~trunc

    return memory


def env_evaluate(
    actor: BaseActor,
    device: torch.device,
    vec_env: VectorEnv,
    num_episodes: int,
) -> tuple[float, float]:
    """Performs an evaluation run using the given actor on the vectorized environment.

    Note that `vec_env.num_envs` must cleanly divide. num_episodes.

    Args:
        actor (BaseActor): actor
        device (torch.device): device
        vec_env (VectorEnv): vec_env
        num_episodes (int): num_episodes

    Returns:
        tuple[float, float]: mean_returns, mean_episode_length
    """
    assert (
        (num_episodes / vec_env.num_envs) % 1.0 == 0
    ), f"`num_episodes` ({num_episodes}) must be clean multiple of {vec_env.num_envs}."
    # start the evaluation loops
    num_valid_steps = 0
    cumulative_rewards = 0.0

    for _ in range(num_episodes // vec_env.num_envs):
        # reset things
        obs, info = vec_env.reset()
        done_envs = np.zeros(vec_env.num_envs, dtype=bool)

        # step for one episode
        while not np.all(done_envs):
            # get an action from the actor
            # this is a tensor
            policy_observation = actor.package_observation(obs, device)
            act = actor.infer(*actor(policy_observation))

            # convert the action to cpu
            act = cpuize(act)

            # step the transition
            next_obs, rew, term, trunc, info = vec_env.step(act)

            # roll observation and record the done envs
            obs = next_obs
            done_envs |= term | trunc

            # record the cumulative rewards and episode length
            cumulative_rewards += np.sum(rew * (1.0 - done_envs))
            num_valid_steps += np.sum(1.0 - done_envs)

    mean_returns = float(cumulative_rewards / num_episodes)
    mean_episode_lengths = float(num_valid_steps / num_episodes)

    return mean_returns, mean_episode_lengths
