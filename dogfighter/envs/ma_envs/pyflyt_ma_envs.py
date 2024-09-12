from typing import Literal

from pettingzoo import ParallelEnv

from dogfighter.envs.ma_envs.base import MAEnvConfig
from dogfighter.envs.ma_envs.wrappers.dogfight_transformer import \
    MAFixedwingDogfightTransformerEnvV2


class PyFlytMAEnvConfig(MAEnvConfig):
    variant: Literal["pyflyt_ma_env"] = "pyflyt_ma_env"  # pyright: ignore

    def instantiate(self) -> ParallelEnv:
        from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2

        if self.env_id == "dogfight":
            if self.env_kwargs.get("flatten_observation"):
                env = MAFixedwingDogfightEnvV2(
                    render_mode=self.render_mode,
                    **self.env_kwargs,
                )
            else:
                env = MAFixedwingDogfightTransformerEnvV2(
                    render_mode=self.render_mode,
                    **self.env_kwargs,
                )

        else:
            raise NotImplementedError

        return env
