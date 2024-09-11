from pydantic import BaseModel

from dogfighter.algorithms import KnownAlgorithmConfigs
from dogfighter.env_interactors import KnownInteractorConfigs
from dogfighter.envs import KnownEnvConfigs
from dogfighter.replay_buffers.replay_buffer import ReplayBufferConfig
from dogfighter.runners import KnownRunnerSettings


class ConfigStack(BaseModel):
    train_env_config: KnownEnvConfigs
    eval_env_config: KnownEnvConfigs
    algorithm_config: KnownAlgorithmConfigs
    memory_config: ReplayBufferConfig
    interactor_config: KnownInteractorConfigs
    runner_settings: KnownRunnerSettings
