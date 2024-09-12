from typing import Literal

import gymnasium as gym

from dogfighter.envs.sa_envs.base import SAEnvConfig


class DMCSAEnvConfig(SAEnvConfig):
    variant: Literal["dmc_sa_env"] = "dmc_sa_env"  # pyright: ignore

    def instantiate(self) -> gym.Env:
        import shimmy
        from gymnasium.wrappers import FlattenObservation

        gym.register_envs(shimmy)

        env = gym.make(
            self.env_id,
            render_mode=self.render_mode,
            **self.env_kwargs,
        )
        env = FlattenObservation(env)

        return env
