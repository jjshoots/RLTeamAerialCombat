import time
from typing import Any, Literal

import numpy as np
import torch
from memorial import ReplayBuffer
from memorial.replay_buffers import FlatReplayBuffer
from pettingzoo import ParallelEnv
from wingman.utils import cpuize, gpuize

from dogfighter.env_interactors.base import (CollectionFunctionProtocol,
                                             DisplayFunctionProtocol,
                                             EnvInteractorConfig,
                                             EvaluationFunctionProtocol,
                                             SupportedEnvTypes)
from dogfighter.models.base.base import Actor


class MLPMAEnvInteractorConfig(EnvInteractorConfig):
    variant: Literal["mlp_ma"] = "mlp_ma"  # pyright: ignore

    def get_collection_fn(self) -> CollectionFunctionProtocol:
        return mlp_ma_env_collect

    def get_evaluation_fn(self) -> EvaluationFunctionProtocol:
        return mlp_ma_env_evaluate

    def get_display_fn(self) -> DisplayFunctionProtocol:
        return mlp_ma_env_display


@torch.no_grad()
def mlp_ma_env_collect(
    actor: Actor,
    env: SupportedEnvTypes,
    memory: ReplayBuffer,
    num_transitions: int,
    use_random_actions: bool,
) -> tuple[ReplayBuffer, dict[str, Any]]:
    """Runs the actor in the multiagent parallel environment and collects transitions.

    This collects `num_transitions` transitions using `num_transitions // env.num_agent` steps.
    Note that it stores the items in the replay buffer in the following order:
    - The first `n` items be for observation. (`n=1` if obs = np.ndarray)
    - The next `n` items be for next observation. (`n=1` if obs = np.ndarray)
    - The next 1 item be for action.
    - The next 1 item be for reward.
    - The next 1 item be for termination.

    Args:
        actor (MlpActor): actor
        env (ParallelEnv): env
        memory (FlatReplayBuffer): memory
        num_transitions (int): num_transitions
        use_random_actions (bool): use_random_actions

    Returns:
        tuple[FlatReplayBuffer, dict[str, Any]]:
    """
    assert isinstance(env, ParallelEnv)
    assert isinstance(memory, FlatReplayBuffer)

    # to record times
    start_time = time.time()

    # list to store memory address references for each transition generated
    transitions = []

    # set to eval and zero grad
    actor.eval()
    actor.zero_grad()

    steps_collected = 0
    while steps_collected < num_transitions:
        # init the first obs, infos
        dict_obs, _ = env.reset()

        # loop interaction
        while env.agents:
            # stack the observation into an array
            stack_obs = np.stack([v for v in dict_obs.values()], axis=0)

            # compute an action depending on whether we're exploring or not
            if use_random_actions:
                # sample an action from the env
                dict_act = {
                    agent: env.action_space(agent).sample() for agent in dict_obs.keys()
                }
                stack_act = np.stack([v for v in dict_act.values()], axis=0)
            else:
                # get an action from the actor
                stack_act = cpuize(
                    actor.sample(*actor(gpuize(stack_obs, actor.device)))[0]
                )
                dict_act = {k: v for k, v in zip(dict_obs.keys(), stack_act)}

            # step the transition
            dict_next_obs, dict_rew, dict_term, dict_trunc, _ = env.step(dict_act)

            # increment step count
            steps_collected += stack_act.shape[0]

            # temporarily store the transitions
            # don't care that it's not contiguous
            # we just want to store memory addresses for now
            transitions.append(
                (
                    stack_obs,
                    stack_act,
                    np.stack([v for v in dict_rew.values()], axis=0)[:, None],
                    np.stack([v for v in dict_term.values()], axis=0)[:, None],
                    np.stack([v for v in dict_next_obs.values()], axis=0),
                )
            )

            # new observation is the next observation
            dict_obs = {
                k: v
                for k, v in dict_next_obs.items()
                if not (dict_term[k] or dict_trunc[k])
            }

    # store stuff in contiguous mem after each episode
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
def mlp_ma_env_evaluate(
    actor: Actor,
    env: SupportedEnvTypes,
    num_episodes: int,
) -> tuple[float, dict[str, Any]]:
    """mlp_ma_env_evaluate.

    Args:
        actor (Actor): actor
        env (SupportedEnvTypes): env
        num_episodes (int): num_episodes

    Returns:
        tuple[float, dict[str, Any]]:
    """
    assert isinstance(env, ParallelEnv)

    # set to eval and zero grad
    actor.eval()
    actor.zero_grad()

    # start the evaluation loops
    num_interactions = 0
    cumulative_reward = 0.0
    num_out_of_bounds = 0
    num_collisions = 0
    num_received_hits = 0

    for _ in range(num_episodes):
        # init the first obs, infos
        dict_obs, dict_info = env.reset()

        while env.agents:
            # convert the dictionary observation into an array and move it to the GPU
            # get an action from the actor, then parse into dictionary
            stack_obs = np.stack([v for v in dict_obs.values()])
            stack_act = cpuize(actor.infer(*actor(gpuize(stack_obs))))
            dict_act = {k: v for k, v in zip(dict_obs.keys(), stack_act)}

            # step a transition, next observation is current observation
            dict_next_obs, dict_rew, dict_term, dict_trunc, dict_info = env.step(
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

        # track statistics
        num_received_hits += sum(
            [i.get("received_hits", 0) for i in dict_info.values()]
        )

    # arrange the results
    # TODO: fix this thing to be generic... somehow
    return_info = dict()
    return_info["mean_episode_interactions"] = float(num_interactions / num_episodes)
    return_info["cumulative_reward"] = cumulative_reward / num_episodes
    return_info["num_out_of_bounds"] = float(num_out_of_bounds / num_episodes)
    return_info["num_collisions"] = float(num_collisions / num_episodes)
    return_info["mean_hits_per_agent"] = float(num_received_hits / num_episodes)
    eval_score = (
        return_info["mean_episode_interactions"] * 0.05
        + return_info["mean_hits_per_agent"] * 10.0
    ) / (return_info["num_collisions"] + return_info["num_out_of_bounds"] + 1)
    return eval_score, return_info


@torch.no_grad()
def mlp_ma_env_display(
    actor: Actor,
    env: SupportedEnvTypes,
) -> None:
    """mlp_ma_env_display.

    Args:
        actor (Actor): actor
        env (SupportedEnvTypes): env

    Returns:
        None:
    """
    assert isinstance(env, ParallelEnv)

    # set to eval and zero grad
    actor.eval()
    actor.zero_grad()

    # init the first obs, infos
    dict_obs, _ = env.reset()

    while env.agents:
        # convert the dictionary observation into an array and move it to the GPU
        # get an action from the actor, then parse into dictionary
        stack_obs = gpuize(np.stack([v for v in dict_obs.values()]), actor.device)
        stack_act = actor.infer(*actor(stack_obs))
        dict_act = {k: v for k, v in zip(dict_obs.keys(), cpuize(stack_act))}

        # step a transition, next observation is current observation
        dict_next_obs, _, dict_term, dict_trunc, _ = env.step(dict_act)
        dict_obs = {
            k: v
            for k, v in dict_next_obs.items()
            if not (dict_term[k] or dict_trunc[k])
        }
