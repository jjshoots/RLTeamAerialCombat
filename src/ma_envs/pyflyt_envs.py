from __future__ import annotations

from pettingzoo import ParallelEnv
from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
from ma_envs.dogfight_transformer import MAFixedwingDogfightTransformerEnvV2


def setup_fixedwing_dogfight_env(
    render_mode: str | None,
    **env_kwargs,
) -> ParallelEnv:

    # define one env
    if env_kwargs.get("flatten"):
        ma_env = MAFixedwingDogfightEnvV2(
            render_mode=render_mode,
            **env_kwargs,
        )
    else:
        ma_env = MAFixedwingDogfightTransformerEnvV2(
            render_mode=render_mode,
            **env_kwargs,
        )

    return ma_env
