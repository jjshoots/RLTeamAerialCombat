import time
from typing import Literal

import numpy as np
import torch
from memorial import ReplayBuffer
from memorial.wrappers import DictReplayBufferWrapper, listed_dict_to_dicted_list
from pettingzoo import ParallelEnv
from wingman.utils import cpuize, nested_gpuize

from dogfighter.env_interactors.base import (
    CollectionFunctionProtocol,
    DisplayFunctionProtocol,
    EnvInteractorConfig,
    EvaluationFunctionProtocol,
    SupportedEnvTypes,
    UpdateInfos,
)
from dogfighter.models.base import Actor


class TransformerMAEnvInteractorConfig(EnvInteractorConfig):
    """TransformerMAEnvInteractorConfig.

    An interactor for multiagent environments with Box space observations.
    """

    variant: Literal["transformer_ma"] = "transformer_ma"  # pyright: ignore

    def get_collection_fn(self) -> CollectionFunctionProtocol:
        """get_collection_fn."""
        return transformer_ma_env_collect

    def get_evaluation_fn(self) -> EvaluationFunctionProtocol:
        """get_evaluation_fn."""
        return transformer_ma_env_evaluate

    def get_display_fn(self) -> DisplayFunctionProtocol:
        """get_display_fn."""
        return transformer_ma_env_display


@torch.no_grad()
def transformer_ma_env_collect(
    actor: Actor,
    env: SupportedEnvTypes,
    memory: ReplayBuffer,
    num_transitions: int,
    use_random_actions: bool,
) -> tuple[ReplayBuffer, UpdateInfos]:
    """Runs the actor in the multiagent parallel environment and collects transitions.

    This collects `num_transitions` transitions using `num_transitions // env.num_agent` steps.
    Note that it stores the items in the replay buffer in the following order:
    - The first `n` items be for observation. (`n=1` if obs = np.ndarray)
    - The next `n` items be for next observation. (`n=1` if obs = np.ndarray)
    - The next 1 item be for action.
    - The next 1 item be for reward.
    - The next 1 item be for termination.

    Args:
        actor (TransformerActor): actor
        env (ParallelEnv): env
        memory (DictReplayBufferWrapper): memory
        num_transitions (int): num_transitions
        use_random_actions (bool): use_random_actions

    Returns:
        tuple[DictReplayBufferWrapper, UpdateInfos]:
    """
    assert isinstance(env, ParallelEnv)
    assert isinstance(memory, DictReplayBufferWrapper)

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
        stack_obs = nested_gpuize(
            listed_dict_to_dicted_list([v for v in dict_obs.values()], stack=True)
        )

        # loop interaction
        while env.agents:
            # compute an action depending on whether we're exploring or not
            if use_random_actions:
                # sample an action from the env
                dict_act = {
                    agent: env.action_space(agent).sample() for agent in dict_obs.keys()
                }
                stack_act = np.stack([v for v in dict_act.values()], axis=0)
            else:
                # get an action from the actor
                stack_act = cpuize(actor.sample(*actor(stack_obs))[0])
                dict_act = {k: v for k, v in zip(dict_obs.keys(), stack_act)}

            # step the transition
            dict_next_obs, dict_rew, dict_term, dict_trunc, _ = env.step(dict_act)

            # stack things
            stack_next_obs = nested_gpuize(
                listed_dict_to_dicted_list(
                    [v for v in dict_next_obs.values()], stack=True
                )
            )
            stack_rew = np.stack([v for v in dict_rew.values()], axis=0)[:, None]
            stack_term = np.stack([v for v in dict_term.values()], axis=0)[:, None]
            stack_trunc = np.stack([v for v in dict_trunc.values()], axis=0)[:, None]

            # temporarily store the transitions
            # don't care that it's not contiguous
            # we just want to store memory addresses for now
            transitions.append(
                (
                    stack_obs,
                    stack_act,
                    stack_rew,
                    stack_term,
                    stack_next_obs,
                )
            )

            # new observation is the next observation
            if np.any(stack_term) or np.any(stack_trunc):
                # we need to reconstruct the observation if any agents term/trunc
                dict_obs = {
                    k: v
                    for k, v in dict_next_obs.items()
                    if not (dict_term[k] or dict_trunc[k])
                }
                if len(dict_obs):
                    stack_obs = nested_gpuize(
                        listed_dict_to_dicted_list(
                            [v for v in dict_obs.values()], stack=True
                        )
                    )
            else:
                # otherwise, we can just reuse the stack
                stack_obs = stack_next_obs

            # increment step count
            steps_collected += stack_act.shape[0]

    # store stuff in contiguous mem after each episode
    memory.push(
        [
            (
                listed_dict_to_dicted_list(items, stack=False)
                if isinstance(items[0], dict)
                else np.concatenate(items, axis=0)
            )
            for items in zip(*transitions)
        ],
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
def transformer_ma_env_evaluate(
    actor: Actor,
    env: SupportedEnvTypes,
    num_episodes: int,
) -> tuple[float, UpdateInfos]:
    """transformer_ma_env_evaluate.

    Args:
        actor (Actor): actor
        env (SupportedEnvTypes): env
        num_episodes (int): num_episodes

    Returns:
        tuple[float, UpdateInfos]:
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
            # convert the dictionary observation into dictionary of arrays, send to GPU
            # get an action from the actor, then parse into dictionary
            stack_obs = listed_dict_to_dicted_list(
                [v for v in dict_obs.values()], stack=True
            )
            stack_act = cpuize(actor.infer(*actor(nested_gpuize(stack_obs))))
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
        return_info["mean_episode_interactions"] * 0.1
        + return_info["mean_hits_per_agent"] * 10.0
    ) / (
        2.0 * return_info["num_collisions"] + 2.0 * return_info["num_out_of_bounds"] + 1
    )
    return eval_score, return_info


@torch.no_grad()
def transformer_ma_env_display(
    actor: Actor,
    env: SupportedEnvTypes,
) -> None:
    """transformer_ma_env_display.

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
        stack_obs = listed_dict_to_dicted_list(
            [v for v in dict_obs.values()], stack=True
        )
        stack_act = cpuize(actor.infer(*actor(nested_gpuize(stack_obs))))
        dict_act = {k: v for k, v in zip(dict_obs.keys(), stack_act)}

        # step a transition, next observation is current observation
        dict_next_obs, _, dict_term, dict_trunc, _ = env.step(dict_act)
        dict_obs = {
            k: v
            for k, v in dict_next_obs.items()
            if not (dict_term[k] or dict_trunc[k])
        }
