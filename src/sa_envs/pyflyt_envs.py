from __future__ import annotations

import gymnasium as gym


def setup_pyflyt_sa_env(env_id: str, render_mode: str | None, **env_kwargs) -> gym.Env:
    from gymnasium.wrappers import RescaleAction
    from PyFlyt.gym_envs import FlattenWaypointEnv

    env = gym.make(
        env_id,
        render_mode=render_mode,
        **env_kwargs,
    )
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    # wrap in flatten if needed
    if "waypoint" in env_id.lower():
        env = FlattenWaypointEnv(env, context_length=1)

    env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    return env
