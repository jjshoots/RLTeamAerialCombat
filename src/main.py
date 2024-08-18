from __future__ import annotations

from signal import SIGINT, signal

from wingman import Wingman
from wingman.utils import shutdown_handler

from dogfighter.bases.base_env_creators import SAVecEnvConfig
from dogfighter.bases.base_env_interactors import (CollectFunctionProtocol,
                                                   EvaluationFunctionProtocol)
from dogfighter.bases.base_replay_buffer import ReplayBufferConfig
from dogfighter.env_interactors.mlp_ma_env_interactor import (
    mlp_ma_env_collect, mlp_ma_env_display, mlp_ma_env_evaluate)
from dogfighter.env_interactors.mlp_sa_vec_env_interactor import (
    mlp_sa_env_display, mlp_sa_vec_env_collect, mlp_sa_vec_env_evaluate)
from dogfighter.env_interactors.transformer_ma_env_interactor import (
    transformer_ma_env_collect, transformer_ma_env_display,
    transformer_ma_env_evaluate)
from dogfighter.runners.synchronous_runner import (SynchronousRunnerSettings,
                                                   run_synchronous)
from setup_algorithms import get_algorithm_config
from setup_envs import (get_mlp_ma_env_config, get_mlp_sa_env_config,
                        get_transformer_ma_env_config)


def train(wm: Wingman) -> None:
    # setup configs
    if wm.cfg.env.variant == "mlp_sa_env":
        train_env_config = SAVecEnvConfig(
            sa_env_config=get_mlp_sa_env_config(wm),
            num_envs=wm.cfg.env.num_envs,
        )
        eval_env_config = SAVecEnvConfig(
            sa_env_config=get_mlp_sa_env_config(wm),
            num_envs=wm.cfg.env.num_envs,
        )
        algorithm_config = get_algorithm_config(wm)
        collect_fn: CollectFunctionProtocol = mlp_sa_vec_env_collect  # pyright: ignore[reportAssignmentType]
        evaluation_fn: EvaluationFunctionProtocol = mlp_sa_vec_env_evaluate  # pyright: ignore[reportAssignmentType]
    elif wm.cfg.env.variant == "mlp_ma_env":
        train_env_config = get_mlp_ma_env_config(wm)
        eval_env_config = get_mlp_ma_env_config(wm)
        algorithm_config = get_algorithm_config(wm)
        collect_fn: CollectFunctionProtocol = mlp_ma_env_collect  # pyright: ignore[reportAssignmentType]
        evaluation_fn: EvaluationFunctionProtocol = mlp_ma_env_evaluate  # pyright: ignore[reportAssignmentType]
    elif wm.cfg.env.variant == "transformer_ma_env":
        train_env_config = get_transformer_ma_env_config(wm)
        eval_env_config = get_transformer_ma_env_config(wm)
        algorithm_config = get_algorithm_config(wm)
        collect_fn: CollectFunctionProtocol = transformer_ma_env_collect  # pyright: ignore[reportAssignmentType]
        evaluation_fn: EvaluationFunctionProtocol = transformer_ma_env_evaluate  # pyright: ignore[reportAssignmentType]
    else:
        raise NotImplementedError

    # perform a run
    run_synchronous(
        wm=wm,
        train_env_config=train_env_config,
        eval_env_config=eval_env_config,
        collect_fn=collect_fn,
        evaluation_fn=evaluation_fn,
        algorithm_config=algorithm_config,
        memory_config=ReplayBufferConfig(
            mem_size=wm.cfg.replay_buffer.mem_size,
            mode=wm.cfg.replay_buffer.mode,
            device=str(wm.device),
            use_dict_wrapper=wm.cfg.replay_buffer.use_dict_wrapper,
            store_on_device=wm.cfg.replay_buffer.store_on_device,
            random_rollover=wm.cfg.replay_buffer.random_rollover,
        ),
        settings=SynchronousRunnerSettings(
            max_transitions=wm.cfg.runner.max_transitions,
            transitions_per_epoch=wm.cfg.runner.transitions_per_epoch,
            transitions_num_exploration=wm.cfg.runner.transitions_num_exploration,
            train_min_transitions=wm.cfg.runner.train_min_transitions,
            eval_num_episodes=wm.cfg.runner.eval_num_episodes,
            eval_transitions_frequency=wm.cfg.runner.eval_transitions_frequency,
        ),
    )


def display(wm: Wingman) -> None:
    if wm.cfg.env.variant == "mlp_ma_env":
        mlp_ma_env_display(
            env=get_mlp_ma_env_config(wm),
            actor=get_algorithm_config(wm).instantiate().actor,
        )
    elif wm.cfg.env.variant == "mlp_vec_env":
        mlp_sa_env_display(
            env=get_mlp_sa_env_config(wm),
            actor=get_algorithm_config(wm).instantiate().actor,
        )
    elif wm.cfg.env.variant == "transformer_ma_env":
        transformer_ma_env_display(
            env=get_transformer_ma_env_config(wm),
            actor=get_algorithm_config(wm).instantiate().actor,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    # wm = Wingman(config_yaml="./configs/quad_dogfight_config.yaml")
    # wm = Wingman(config_yaml="./configs/quadx_pole_balance_config.yaml")
    wm = Wingman(config_yaml="./configs/dual_dogfight_transformer_config.yaml")
    # wm = Wingman(config_yaml="./configs/dual_dogfight_mlp_config.yaml")

    if wm.cfg.mode.train:
        train(wm)
    elif wm.cfg.mode.display:
        display(wm)
    else:
        print("So this is life now.")
