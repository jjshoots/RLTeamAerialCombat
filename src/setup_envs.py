from __future__ import annotations

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, VectorEnv
from pettingzoo import ParallelEnv
from wingman import Wingman

from ma_envs.pyflyt_envs import setup_fixedwing_dogfight_env
from sa_envs.dmc_envs import setup_dmc_sa_env
from sa_envs.pyflyt_envs import setup_pyflyt_sa_env


def setup_sa_vec_environment(wm: Wingman) -> VectorEnv:
    vec_env = AsyncVectorEnv(
        [lambda _=i: setup_sa_environment(wm) for i in range(wm.cfg.num_envs)]
    )
    return vec_env


def setup_sa_environment(wm: Wingman) -> gym.Env:
    if wm.cfg.env.name.startswith("PyFlyt"):
        env = setup_pyflyt_sa_env(
            wm.cfg.env.name,
            render_mode="human" if wm.cfg.mode.display or wm.cfg.mode.render else None,
            flight_mode=-1,
        )
    elif wm.cfg.env.name.startswith("dm_control"):
        env = setup_dmc_sa_env(
            wm.cfg.env.name,
            render_mode="human" if wm.cfg.mode.display or wm.cfg.mode.render else None,
        )
    else:
        raise NotImplementedError

        # record observation and action space shapes
    if not getattr(wm.cfg, "obs_size", None):
        wm.cfg.model.obs_size = env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
    if not getattr(wm.cfg, "act_size", None):
        wm.cfg.model.act_size = env.action_space.shape[0]  # pyright: ignore[reportOptionalSubscript]

    return env


def setup_ma_environment(wm: Wingman) -> ParallelEnv:
    if wm.cfg.env.name == "dogfight":
        env = setup_fixedwing_dogfight_env(
            render_mode="human" if wm.cfg.mode.display or wm.cfg.mode.render else None,
            env_kwargs=vars(wm.cfg.env.kwargs),
        )
    else:
        raise NotImplementedError

    if wm.cfg.env.flatten:
        # record observation and action space shapes
        if not getattr(wm.cfg, "src_size", None):
            wm.cfg.model.obs_size = env.observation_space(0)["src"].feature_space.shape[
                0
            ]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
        if not getattr(wm.cfg, "tgt_size", None):
            wm.cfg.model.act_size = env.action_space(0)["tgt"].feature_space.shape[0]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
        if not getattr(wm.cfg, "act_size", None):
            wm.cfg.model.act_size = env.action_space(0).shape[0]  # pyright: ignore[reportOptionalSubscript]
    else:
        # record observation and action space shapes
        if not getattr(wm.cfg, "obs_size", None):
            wm.cfg.model.obs_size = env.observation_space(0).shape[0]  # pyright: ignore[reportOptionalSubscript]
        if not getattr(wm.cfg, "act_size", None):
            wm.cfg.model.act_size = env.action_space(0).shape[0]  # pyright: ignore[reportOptionalSubscript]

    return env
