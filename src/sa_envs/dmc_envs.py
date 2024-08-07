from __future__ import annotations

import gymnasium as gym


def setup_dmc_sa_env(env_id: str, render_mode: str | None, **env_kwargs) -> gym.Env:
    import shimmy
    from gymnasium.wrappers import FlattenObservation, RescaleAction

    gym.register_envs(shimmy)

    env = gym.make(
        env_id,
        render_mode=render_mode,
        **env_kwargs,
    )
    env = FlattenObservation(env)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    return env
