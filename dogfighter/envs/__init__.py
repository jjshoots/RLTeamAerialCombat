from typing import Union

from dogfighter.envs.base import MAEnvConfig, SAVecEnvConfig

KnownEnvConfigs = Union[
    SAVecEnvConfig,
    MAEnvConfig,
]
