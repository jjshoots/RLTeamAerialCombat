from __future__ import annotations

from pettingzoo import ParallelEnv


def setup_fixedwing_dogfight_env(
    render_mode: str | None,
    flatten_observation: bool,
    **env_kwargs,
) -> ParallelEnv:
    from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2

    # define one env
    ma_env = MAFixedwingDogfightEnvV2(
        render_mode=render_mode,
        flatten_observation=flatten_observation,
        **env_kwargs,
    )

    return ma_env
