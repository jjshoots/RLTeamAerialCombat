from typing import Union

from dogfighter.envs.sa_envs.dmc_sa_env import DMCSAEnvConfig
from dogfighter.envs.sa_envs.pyflyt_sa_envs import PyFlytSAEnvConfig

KnownSAEnvConfigs = Union[
    DMCSAEnvConfig,
    PyFlytSAEnvConfig,
]
