from __future__ import annotations

import gymnasium as gym

from dogfighter.bases.base_env_creators import SAEnvConfig


class DMCSAEnvConfig(SAEnvConfig):
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
