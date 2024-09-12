from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from pydantic import BaseModel, StrictInt

from dogfighter.envs.sa_envs import KnownSAEnvConfigs


class SAVecEnvConfig(BaseModel):
    """Initializer for single agent vector environments."""

    sa_env_config: KnownSAEnvConfigs
    num_envs: StrictInt

    def instantiate(self) -> VectorEnv:
        if self.num_envs == 1:
            return SyncVectorEnv(
                [lambda _=i: self.sa_env_config.instantiate() for i in range(self.num_envs)]
            )
        elif self.num_envs > 1:
            return AsyncVectorEnv(
                [lambda _=i: self.sa_env_config.instantiate() for i in range(self.num_envs)]
            )
        else:
            raise ValueError(f"`num_envs` ({self.num_envs}) must be more than 1")
