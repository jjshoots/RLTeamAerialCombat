from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2


class MAFixedwingDogfightTransformerEnvV2(MAFixedwingDogfightEnvV2):
    """MAFixedwingDogfightTransformerEnv."""

    def __init__(self, **kwargs) -> None:
        """Normal dogfight env in non-flatten mode.

        This alters the observation space to be transformer friendly,
        "others" observation is now "src" and a gymnasium.spaces.Sequence.
        "self" observation is now "tgt" and a gymnasium.spaces.Sequence.
        """
        super().__init__(**kwargs)
        self._transformer_space = spaces.Dict(
            dict(
                src=super().observation_space(0)["others"],
                tgt=spaces.Sequence(
                    space=super().observation_space(0)["self"],
                ),
            )
        )

    def observation_space(self, agent: Any) -> spaces.Dict:
        return self._transformer_space

    def reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # TODO: HERE
        obs, infos = super().reset(seed=seed, options=options)
        return obs, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        # TODO: HERE
        obs, rew, term, trunc, infos = super().step(actions=actions)
        return obs, rew, term, trunc, infos
