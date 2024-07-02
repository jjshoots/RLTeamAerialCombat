from __future__ import annotations

import gymnasium as gym
import torch
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import rescale_action
from PyFlyt.gym_envs import FlattenWaypointEnv
from wingman import ReplayBuffer, Wingman

from dogfighter.algorithms import CCGE
from dogfighter.algorithms.ccge import CCGEParams
from dogfighter.models.mlp import MlpActor, MlpEnvParams, MlpQUEnsemble
from dogfighter.models.mlp.mlp_bases import MlpModelParams


def setup_replay_buffer(wm: Wingman) -> ReplayBuffer:
    return ReplayBuffer(
        mem_size=wm.cfg.buffer_size,
        mode=wm.cfg.replay_buffer_mode,
        device=wm.device,
        store_on_device=wm.cfg.replay_buffer_store_on_device,
        random_rollover=wm.cfg.random_rollover,
    )


def setup_vector_environment(wm: Wingman) -> AsyncVectorEnv:
    return AsyncVectorEnv(
        [lambda i=i: setup_single_environment(wm) for i in range(wm.cfg.num_envs)]
    )


def setup_single_environment(wm: Wingman) -> gym.Env:
    # define one env
    env = gym.make(
        wm.cfg.env_name,
        render_mode="human" if wm.cfg.display or wm.cfg.render else None,
        flight_mode=-1,
    )
    env = rescale_action.RescaleAction(env, min_action=-1.0, max_action=1.0)

    # wrap in flatten if needed
    if "waypoint" in wm.cfg.env_name.lower():
        env = FlattenWaypointEnv(env, context_length=1)

    # record observation space shape
    wm.cfg.obs_size = env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
    wm.cfg.act_size = env.action_space.shape[0]  # pyright: ignore[reportOptionalSubscript]

    return env


def setup_algorithm(wm: Wingman) -> CCGE:
    # define some params
    env_params = MlpEnvParams(
        obs_size=wm.cfg.obs_size,
        act_size=wm.cfg.act_size,
    )
    model_params = MlpModelParams(
        qu_num_ensemble=wm.cfg.qu_num_ensemble,
        embed_dim=wm.cfg.embed_dim,
    )
    algorithm_params = CCGEParams()

    # define the algorithm, conditionally jit
    alg = CCGE(
        actor_type=MlpActor,
        critic_type=MlpQUEnsemble,
        env_params=env_params,
        model_params=model_params,
        algorithm_params=algorithm_params,
        device=torch.device(wm.device),
    )
    if not wm.cfg.debug:
        torch.compile(alg)

    # get latest weight files
    has_weights, model_file, _ = wm.get_weight_files()
    if has_weights:
        # load the model
        alg.load_state_dict(
            torch.load(model_file, map_location=torch.device(wm.cfg.device))
        )

    return alg
