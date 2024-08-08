from __future__ import annotations

from typing import Any, Literal

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
        assert not kwargs["flatten"]
        super().__init__(**kwargs)
        self._num_agents = super().num_possible_agents
        self._transformer_space = spaces.Dict(
            dict(
                src=super().observation_space(0)["others"],  # Sequence
                tgt=spaces.Sequence(
                    space=super().observation_space(0)["self"],
                ),
                src_mask=spaces.Box(0, 1, shape=(self._num_agents - 1)),
                tgt_mask=spaces.Box(0, 1, shape=(self._num_agents - 1)),
            )
        )

        # the tgt mask is constant
        self._base_src = np.zeros(
            (
                self._num_agents - 1,
                self.observation_space(0)["src"].feature_space.shape[0],  # pyright: ignore[reportOptionalSubscript]
            ),
            dtype=np.float64,
        )
        self._base_src_mask = np.zeros((self._num_agents - 1,), dtype=bool)
        self._tgt_mask = np.array([True], dtype=bool)

    def observation_space(
        self, agent: Any
    ) -> dict[
        Literal["src", "tgt", "src_mask", "tgt_mask"],
        spaces.Sequence | spaces.Box,
    ]:
        return self._transformer_space  # pyright: ignore[reportReturnType]

    def _convert_obs(
        self,
        obs: dict[
            str,
            dict[Literal["self", "others"], np.ndarray],
        ],
    ) -> dict[
        str,
        dict[
            Literal["src", "tgt", "src_mask", "tgt_mask"],
            np.ndarray,
        ],
    ]:
        transformer_obs = {}
        for agent_id, agent_obs in obs.items():
            # deal with src
            src = self._base_src.copy()
            src[: len(agent_obs["others"])] = agent_obs["others"]
            src_mask = self._base_src_mask.copy()
            src_mask[: len(agent_obs["others"])] = True

            # deal with target
            tgt = np.expand_dims(agent_obs["self"], axis=0)
            tgt_mask = self._tgt_mask.copy()

            # stack everything
            transformer_obs[agent_id] = {}
            transformer_obs[agent_id]["src"] = src
            transformer_obs[agent_id]["src_mask"] = src_mask
            transformer_obs[agent_id]["tgt"] = tgt
            transformer_obs[agent_id]["tgt_mask"] = tgt_mask
        return transformer_obs

    def reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, infos = super().reset(seed=seed, options=options)
        return self._convert_obs(obs), infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        obs, rew, term, trunc, infos = super().step(actions=actions)
        return self._convert_obs(obs), rew, term, trunc, infos
