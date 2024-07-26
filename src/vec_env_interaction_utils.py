import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from wingman import Wingman
from wingman.replay_buffer import ReplayBuffer
from wingman.utils import cpuize, gpuize

from dogfighter.models.bases import BaseActor
from setup_utils import setup_algorithm, setup_single_environment


@torch.no_grad()
def vec_env_collect_to_memory(
    actor: BaseActor,
    env: VectorEnv,
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
        env (VectorEnv): env
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
    obs, info = env.reset()
    non_reset_envs = np.ones(env.num_envs, dtype=bool)

    for _ in range(num_transitions // env.num_envs):
        # compute an action depending on whether we're exploring or not
        if random_actions:
            # sample an action from the env
            act = env.action_space.sample()
        else:
            # get an action from the actor
            act, _ = actor.sample(*actor(obs))

        # step the transition
        next_obs, rew, term, trunc, info = env.step(act)

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
    env: VectorEnv,
    num_episodes: int,
) -> dict[Literal["eval_perf", "mean_episode_length"], float]:
    """Performs an evaluation run using the given actor on the vectorized environment.

    Note that `env.num_envs` must cleanly divide num_episodes.

    Args:
        actor (BaseActor): actor
        env (VectorEnv): env
        num_episodes (int): num_episodes

    Returns:
        dict[Literal["eval_perf", "mean_episode_length"], float]:
    """
    assert (
        (num_episodes / env.num_envs) % 1.0 == 0
    ), f"`num_episodes` ({num_episodes}) must be clean multiple of {env.num_envs}."
    # start the evaluation loops
    num_valid_steps = 0
    cumulative_rewards = 0.0

    for _ in range(num_episodes // env.num_envs):
        # reset things
        obs, _ = env.reset()
        done_envs = np.zeros(env.num_envs, dtype=bool)

        # step for one episode
        while not np.all(done_envs):
            # get an action from the actor
            # this is a tensor
            act = actor.infer(*actor(obs))

            # convert the action to cpu
            act = cpuize(act)

            # step the transition
            next_obs, rew, term, trunc, _ = env.step(act)

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


def vec_env_render_gif(wm: Wingman) -> Path:
    import imageio.v3 as iio

    frames = []

    # setup the environment and actor
    env = setup_single_environment(wm)
    actor = setup_algorithm(wm).actor

    term, trunc = False, False
    obs, _ = env.reset()

    # step for one episode
    while not term and not trunc:
        # get an action from the actor
        obs = gpuize(obs, device=wm.device).unsqueeze(0)
        act = actor.infer(*actor(obs))
        act = cpuize(act.squeeze(0))

        # step the transition
        next_obs, _, term, trunc, _ = env.step(act)

        # new observation is the next observation
        obs = next_obs

        # for gif
        frames.append(env.render())

    gif_path = Path("/tmp") / Path(
        "gif"
        # "".join(random.choices(string.ascii_letters + string.digits, k=8))
    ).with_suffix(".gif")

    iio.imwrite(
        gif_path,
        frames,
        fps=30,
    )

    return gif_path.absolute()
