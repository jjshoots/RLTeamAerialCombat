from wingman import Wingman

from dogfighter.algorithms.base import AlgorithmConfig
from dogfighter.env_interactors.base import EnvInteractorConfig
from dogfighter.env_interactors.mlp_ma_env_interactor import \
    MLPMAEnvInteractorConfig
from dogfighter.env_interactors.mlp_sa_vec_env_interactor import \
    MLPSAEnvInteractorConfig
from dogfighter.env_interactors.transformer_ma_env_interactor import \
    TransformerMAEnvInteractorConfig
from dogfighter.envs.base import MAEnvConfig, SAVecEnvConfig
from dogfighter.replay_buffers.replay_buffer import ReplayBufferConfig
from dogfighter.runners.asynchronous_runner.base import \
    AsynchronousRunnerSettings
from dogfighter.runners.synchronous_runner import SynchronousRunnerSettings
from setup_algorithms import get_algorithm_config
from setup_envs import (get_mlp_ma_env_config, get_mlp_sa_env_config,
                        get_transformer_ma_env_config)


def get_all_configs(
    wm: Wingman,
) -> tuple[
    SAVecEnvConfig | MAEnvConfig,
    SAVecEnvConfig | MAEnvConfig,
    EnvInteractorConfig,
    AlgorithmConfig,
    ReplayBufferConfig,
    SynchronousRunnerSettings | AsynchronousRunnerSettings,
]:
    # get env and interactors
    if wm.cfg.env.variant == "mlp_sa_env":
        train_env_config = SAVecEnvConfig(
            sa_env_config=get_mlp_sa_env_config(wm),
            num_envs=wm.cfg.env.num_envs,
        )
        eval_env_config = SAVecEnvConfig(
            sa_env_config=get_mlp_sa_env_config(wm),
            num_envs=wm.cfg.env.num_envs,
        )
        interactor_config = MLPSAEnvInteractorConfig()
    elif wm.cfg.env.variant == "mlp_ma_env":
        train_env_config = get_mlp_ma_env_config(wm)
        eval_env_config = get_mlp_ma_env_config(wm)
        interactor_config = MLPMAEnvInteractorConfig()
    elif wm.cfg.env.variant == "transformer_ma_env":
        train_env_config = get_transformer_ma_env_config(wm)
        eval_env_config = get_transformer_ma_env_config(wm)
        interactor_config = TransformerMAEnvInteractorConfig()
    else:
        raise NotImplementedError

    # algorithm and memory
    algorithm_config = get_algorithm_config(wm)
    memory_config = ReplayBufferConfig(
        mem_size=wm.cfg.replay_buffer.mem_size,
        mode=wm.cfg.replay_buffer.mode,
        device=str(wm.device),
        use_dict_wrapper=wm.cfg.replay_buffer.use_dict_wrapper,
        store_on_device=wm.cfg.replay_buffer.store_on_device,
        random_rollover=wm.cfg.replay_buffer.random_rollover,
    )

    # runner
    if wm.cfg.runner.variant == "async":
        runner_settings = AsynchronousRunnerSettings(**wm.cfg.runner.to_dict())
    elif wm.cfg.runner.variant == "sync":
        runner_settings = SynchronousRunnerSettings(**wm.cfg.runner.to_dict())
    else:
        raise NotImplementedError

    return (
        train_env_config,
        eval_env_config,
        interactor_config,
        algorithm_config,
        memory_config,
        runner_settings,
    )
