from typing import Literal

import gymnasium as gym

from dogfighter.envs.sa_envs.base import SAEnvConfig


class GymnasiumSAEnvConfig(SAEnvConfig):
    variant: Literal["gymnasium_sa_env"] = "gymnasium_sa_env"  # pyright: ignore

    def instantiate(self) -> gym.Env:
        env = gym.make(
            self.env_id,
            render_mode=self.render_mode,
            **self.env_kwargs,
        )

        return env
