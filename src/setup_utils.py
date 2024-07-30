from __future__ import annotations

import gymnasium as gym
import torch
from gymnasium.vector import AsyncVectorEnv, VectorEnv
from gymnasium.wrappers import rescale_action
from pettingzoo import ParallelEnv
from PyFlyt.gym_envs import FlattenWaypointEnv
from PyFlyt.pz_envs import MAFixedwingDogfightEnv
from wingman import Wingman
from wingman.replay_buffer import ReplayBuffer

from dogfighter.algorithms.ccge import CCGEConfig
from dogfighter.bases.base_algorithm import Algorithm
from dogfighter.bases.base_replay_buffer import ReplayBufferConfig
from dogfighter.env_utils.vec_env_gpuize_wrapper import VecEnvGpuizeWrapper
from dogfighter.models.mlp.mlp_actor import MlpActorConfig
from dogfighter.models.mlp.mlp_qu_network import MlpQUNetworkConfig


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


def setup_vector_environment(wm: Wingman) -> VectorEnv:
    # make the vec env
    vec_env = AsyncVectorEnv(
        [lambda _=i: setup_single_environment(wm) for i in range(wm.cfg.num_envs)]
    )

    # wrapper to convert the observation to tensors
    vec_env = VecEnvGpuizeWrapper(vec_env, wm.device)

    return vec_env


def setup_ma_environment(wm: Wingman) -> ParallelEnv:
    # define one env
    ma_env = MAFixedwingDogfightEnv(
        team_size=wm.cfg.team_size,
        render_mode="human" if wm.cfg.display else None,
        flatten_observation=True,
    )

    # record observation space shape
    wm.cfg.obs_size = ma_env.observation_space(0).shape[0]
    wm.cfg.act_size = ma_env.action_space(0).shape[0]

    return ma_env


def setup_replay_buffer(wm: Wingman) -> ReplayBuffer:
    return ReplayBufferConfig(
        mem_size=wm.cfg.buffer_size,
        mode=wm.cfg.replay_buffer_mode,
        device=str(wm.device),
        store_on_device=wm.cfg.replay_buffer_store_on_device,
        random_rollover=wm.cfg.random_rollover,
    ).instantiate()


def setup_algorithm(wm: Wingman) -> Algorithm:
    alg = CCGEConfig(
        device=str(wm.device),
        actor_config=MlpActorConfig(
            obs_size=wm.cfg.obs_size,
            act_size=wm.cfg.act_size,
            embed_dim=wm.cfg.embed_dim,
        ),
        qu_config=MlpQUNetworkConfig(
            obs_size=wm.cfg.obs_size,
            act_size=wm.cfg.act_size,
            embed_dim=wm.cfg.embed_dim,
        ),
        target_entropy=(-wm.cfg.act_size),
    ).instantiate()

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
