import time
from pprint import pformat
from typing import Literal

import numpy as np
import torch
from pettingzoo import ParallelEnv
from wingman.replay_buffer import ReplayBuffer
from wingman.utils import cpuize, gpuize

from dogfighter.models.bases import BaseActor


@torch.no_grad()
def ma_env_collect_to_memory(
    actor: BaseActor,
    ma_env: ParallelEnv,
    memory: ReplayBuffer,
    num_transitions: int,
    random_actions: bool,
) -> tuple[ReplayBuffer, dict[Literal["interactions_per_second"], float]]:
    """Runs the actor in the multiagent parallel environment and collects transitions.

    This collects `num_transitions` transitions using `num_transitions // ma_env.num_agent` steps.
    Note that it stores the items in the replay buffer in the following order:
    - The first `n` items be for observation. (`n=1` if obs = np.ndarray)
    - The next `n` items be for next observation. (`n=1` if obs = np.ndarray)
    - The next 1 item be for action.
    - The next 1 item be for reward.
    - The next 1 item be for termination.

    Args:
        actor (BaseActor): actor
        ma_env (ParallelEnv): ma_env
        memory (ReplayBuffer): memory
        num_transitions (int): num_transitions
        random_actions (bool): random_actions

    Returns:
        tuple[ReplayBuffer, dict[Literal["interactions_per_second"], float]]:
    """
    # to record times
    start_time = time.time()

    # set to eval and zero grad
    actor.eval()
    actor.zero_grad()

    steps_collected = 0
    while steps_collected < num_transitions:
        # init the first obs, infos
        dict_obs, _ = ma_env.reset()

        while ma_env.agents:
            # stack the observation into an array
            stack_obs = torch.stack([gpuize(v) for v in dict_obs.values()], dim=0)

            # compute an action depending on whether we're exploring or not
            if random_actions:
                # sample an action from the env
                dict_act = {
                    agent: ma_env.action_space(agent).sample()
                    for agent in dict_obs.keys()
                }
                stack_act = torch.stack([gpuize(v) for v in dict_act.values()], dim=0)
            else:
                # get an action from the actor
                stack_act, _ = actor.sample(*actor(stack_obs))
                dict_act = {k: v for k, v in zip(dict_obs.keys(), cpuize(stack_act))}

            # step the transition
            dict_next_obs, dict_rew, dict_term, dict_trunc, _ = ma_env.step(dict_act)

            # increment step count
            steps_collected += stack_obs.shape[0]

            # store stuff in mem
            memory.push(
                [
                    stack_obs,
                    stack_act,
                    torch.stack([gpuize(v) for v in dict_rew.values()], dim=0)[:, None],
                    torch.stack([gpuize(v) for v in dict_term.values()], dim=0)[:, None],
                    torch.stack([gpuize(v) for v in dict_next_obs.values()], dim=0),
                ],
                bulk=True,
            )

            # new observation is the next observation
            dict_obs = {
                k: v
                for k, v in dict_next_obs.items()
                if not (dict_term[k] or dict_trunc[k])
            }

    # print some recordings
    total_time = time.time() - start_time
    interaction_per_second = num_transitions / total_time
    print(f"Collect Stats: {total_time:.2f}s @ {interaction_per_second} t/s.")

    # return the replay buffer and some information
    collect_info: dict[Literal["interactions_per_second"], float] = dict()
    collect_info["interactions_per_second"] = interaction_per_second
    return memory, collect_info


def ma_env_evaluate(
    actor: BaseActor,
    ma_env: ParallelEnv,
    num_episodes: int,
) -> dict[
    Literal[
        "mean_episode_interactions",
        "cumulative_reward",
        "num_out_of_bounds",
        "num_collisions",
        "mean_hits_per_agent",
    ],
    float,
]:
    """ma_env_evaluate.

    Args:
        actor (BaseActor): actor
        ma_env (ParallelEnv): ma_env
        num_episodes (int): num_episodes

    Returns:
        dict[
            Literal[
                "mean_episode_interactions",
                "cumulative_reward",
                "num_out_of_bounds",
                "num_collisions",
                "mean_hits_per_agent",
            ],
            float
        ]:
    """
    # start the evaluation loops
    num_interactions = 0
    cumulative_reward = 0.0
    num_out_of_bounds = 0
    num_collisions = 0
    num_received_hits = 0

    for _ in range(num_episodes):
        # init the first obs, infos
        dict_obs, dict_info = ma_env.reset()

        while ma_env.agents:
            # convert the dictionary observation into an array and move it to the GPU
            # get an action from the actor, then parse into dictionary
            stack_obs = gpuize(np.stack([v for v in dict_obs.values()]))
            stack_act = actor.infer(*actor(stack_obs))
            dict_act = {k: v for k, v in zip(dict_obs.keys(), cpuize(stack_act))}

            # step a transition, next observation is current observation
            dict_next_obs, dict_rew, dict_term, dict_trunc, dict_info = ma_env.step(
                dict_act
            )
            dict_obs = {
                k: v
                for k, v in dict_next_obs.items()
                if not (dict_term[k] or dict_trunc[k])
            }

            # track statistics
            num_interactions += 1
            cumulative_reward += sum([rew for rew in dict_rew.values()])
            for agent_info in dict_info.values():
                num_out_of_bounds += agent_info.get("out_of_bounds", 0)
                num_collisions += agent_info.get("collision", 0)
                num_received_hits += agent_info.get("received_hits", 0)

    # arrange the results
    info: dict[
        Literal[
            "mean_episode_interactions",
            "cumulative_reward",
            "num_out_of_bounds",
            "num_collisions",
            "mean_hits_per_agent",
        ],
        float,
    ] = dict()
    info["mean_episode_interactions"] = float(num_interactions / num_episodes)
    info["cumulative_reward"] = cumulative_reward / num_episodes
    info["num_out_of_bounds"] = float(num_out_of_bounds / num_episodes)
    info["num_collisions"] = float(num_collisions / num_episodes)
    info["mean_hits_per_agent"] = num_received_hits / num_episodes
    print("Evaluation Stats:\n" f"{pformat(info, indent=2)}\n")
    return info
