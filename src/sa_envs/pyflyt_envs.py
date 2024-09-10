import gymnasium as gym
from pettingzoo import ParallelEnv

from dogfighter.envs.base import MAEnvConfig, SAEnvConfig


class PyFlytSAEnvConfig(SAEnvConfig):
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


class PyFlytMAEnvConfig(MAEnvConfig):
    def instantiate(self) -> ParallelEnv:
        from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2

        from ma_envs.dogfight_transformer import \
            MAFixedwingDogfightTransformerEnvV2

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
