from typing import Union

from dogfighter.envs.ma_envs import KnownMAEnvConfigs
from dogfighter.envs.sa_envs import KnownSAEnvConfigs
from dogfighter.envs.sa_vec_env import SAVecEnvConfig

KnownEnvConfigs = Union[
    KnownMAEnvConfigs,
    KnownSAEnvConfigs,
    SAVecEnvConfig,
]
