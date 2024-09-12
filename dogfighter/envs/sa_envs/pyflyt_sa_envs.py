from typing import Literal

import gymnasium as gym

from dogfighter.envs.sa_envs.base import SAEnvConfig


class PyFlytSAEnvConfig(SAEnvConfig):
    variant: Literal["pyflyt_sa_env"] = "pyflyt_sa_env"  # pyright: ignore

    def instantiate(self) -> gym.Env:
        from gymnasium.wrappers import RescaleAction
        from PyFlyt import gym_envs as pf_envs
        from PyFlyt.gym_envs import FlattenWaypointEnv

        gym.register_envs(pf_envs)

        env = gym.make(
            self.env_id,
            render_mode=self.render_mode,
            **self.env_kwargs,
        )

        # wrap in flatten if needed
        if "waypoint" in self.env_id.lower():
            env = FlattenWaypointEnv(env, context_length=1)

        env = RescaleAction(env, min_action=-1.0, max_action=1.0)

        return env
