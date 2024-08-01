from __future__ import annotations

from signal import SIGINT, signal

from wingman import Wingman
from wingman.utils import shutdown_handler

from dogfighter.bases.base_env_interactors import (CollectFunctionProtocol,
                                                   EvaluationFunctionProtocol)
from dogfighter.env_interactors.ma_env_interactor import (ma_env_collect,
                                                          ma_env_display,
                                                          ma_env_evaluate)
from dogfighter.env_interactors.sa_vec_env_interactor import (
    sa_env_display, sa_vec_env_collect, sa_vec_env_evaluate)
from dogfighter.runners.synchronous_runner import (SynchronousRunnerSettings,
                                                   run_synchronous)
from setup_utils import (setup_algorithm, setup_ma_environment,
                         setup_replay_buffer, setup_single_environment,
                         setup_vector_environment)


def train(wm: Wingman) -> None:
    # setup configs
    if wm.cfg.env_type == "ma_env":
        train_env = setup_ma_environment(wm)
        eval_env = setup_ma_environment(wm)
        algorithm = setup_algorithm(wm)
        memory = setup_replay_buffer(wm)
        collect_fn: CollectFunctionProtocol = ma_env_collect  # pyright: ignore[reportAssignmentType]
        evaluation_fn: EvaluationFunctionProtocol = ma_env_evaluate  # pyright: ignore[reportAssignmentType]
    elif wm.cfg.env_type == "vec_env":
        train_env = setup_vector_environment(wm)
        eval_env = setup_vector_environment(wm)
        algorithm = setup_algorithm(wm)
        memory = setup_replay_buffer(wm)
        collect_fn: CollectFunctionProtocol = sa_vec_env_collect  # pyright: ignore[reportAssignmentType]
        evaluation_fn: EvaluationFunctionProtocol = sa_vec_env_evaluate  # pyright: ignore[reportAssignmentType]
    else:
        raise NotImplementedError

    # perform a run
    run_synchronous(
        wm=wm,
        train_env=train_env,
        eval_env=eval_env,
        collect_fn=collect_fn,
        evaluation_fn=evaluation_fn,
        algorithm=algorithm,
        memory=memory,
        settings=SynchronousRunnerSettings(
            max_transitions=wm.cfg.runner.max_transitions,
            transitions_per_epoch=wm.cfg.runner.transitions_per_epoch,
            transitions_num_exploration=wm.cfg.runner.transitions_num_exploration,
            train_min_transitions=wm.cfg.runner.train_min_transitions,
            train_steps_per_epoch=wm.cfg.runner.train_steps_per_epoch,
            train_batch_size=wm.cfg.runner.train_batch_size,
            eval_num_episodes=wm.cfg.runner.eval_num_episodes,
            eval_transitions_frequency=wm.cfg.runner.eval_transitions_frequency,
        ),
    )


def display(wm: Wingman) -> None:
    if wm.cfg.env_type == "ma_env":
        ma_env_display(
            env=setup_ma_environment(wm),
            actor=setup_algorithm(wm).actor,
        )
    elif wm.cfg.env_type == "vec_env":
        sa_env_display(
            env=setup_single_environment(wm),
            actor=setup_algorithm(wm).actor,
        )


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    # wm = Wingman(config_yaml="./configs/quad_dogfight_config.yaml")
    wm = Wingman(config_yaml="./configs/dual_dogfight_config.yaml")
    # wm = Wingman(config_yaml="./configs/quadx_pole_balance_config.yaml")
    # wm = Wingman(config_yaml="./configs/dmc_cheetah_run_config.yaml")

    if wm.cfg.mode.train:
        train(wm)
    elif wm.cfg.mode.display:
        display(wm)
    else:
        print("So this is life now.")
