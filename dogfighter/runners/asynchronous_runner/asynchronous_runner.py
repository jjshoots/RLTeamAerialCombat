from wingman import Wingman

from dogfighter.algorithms.base import AlgorithmConfig
from dogfighter.env_interactors.base import EnvInteractorConfig
from dogfighter.envs.base import MAEnvConfig, SAVecEnvConfig
from dogfighter.replay_buffers.replay_buffer import ReplayBufferConfig
from dogfighter.runners.asynchronous_runner.base import \
    AsynchronousRunnerSettings
from dogfighter.runners.asynchronous_runner.trainer import run_train
from dogfighter.runners.asynchronous_runner.workers import (run_collection,
                                                            run_evaluation)
from dogfighter.runners.synchronous_runner import SynchronousRunnerSettings


def run_asynchronous(
    wm: Wingman,
    train_env_config: SAVecEnvConfig | MAEnvConfig,
    eval_env_config: SAVecEnvConfig | MAEnvConfig,
    algorithm_config: AlgorithmConfig,
    memory_config: ReplayBufferConfig,
    interactor_config: EnvInteractorConfig,
    settings: SynchronousRunnerSettings | AsynchronousRunnerSettings,
) -> None:
    assert isinstance(settings, AsynchronousRunnerSettings)

    # assert that we are async
    assert wm.cfg.runner.variant == "async"

    # depending on the mode, run eval or collect
    if wm.cfg.runner.mode == "trainer":
        run_train(
            wm=wm,
            algorithm_config=algorithm_config,
            memory_config=memory_config,
            settings=settings,
        )
    elif wm.cfg.runner.mode == "worker":
        if wm.cfg.runner.worker.task == "collect":
            run_collection(
                train_env_config=train_env_config,
                algorithm_config=algorithm_config,
                memory_config=memory_config,
                interactor_config=interactor_config,
                settings=settings,
            )
        elif wm.cfg.runner.worker.task == "eval":
            run_evaluation(
                eval_env_config=eval_env_config,
                algorithm_config=algorithm_config,
                interactor_config=interactor_config,
                settings=settings,
            )
        else:
            raise NotImplementedError
