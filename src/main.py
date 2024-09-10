from pathlib import Path
from signal import SIGINT, signal

from wingman import Wingman
from wingman.utils import shutdown_handler

from dogfighter.env_interactors.mlp_ma_env_interactor import (
    MLPMAEnvInteractorConfig, mlp_ma_env_display)
from dogfighter.env_interactors.mlp_sa_vec_env_interactor import (
    MLPSAEnvInteractorConfig, mlp_sa_env_display)
from dogfighter.env_interactors.transformer_ma_env_interactor import (
    TransformerMAEnvInteractorConfig, transformer_ma_env_display)
from dogfighter.envs.base import SAVecEnvConfig
from dogfighter.replay_buffers.replay_buffer import ReplayBufferConfig
from dogfighter.runners.asynchronous_runner import (AsynchronousRunnerSettings,
                                                    run_asynchronous)
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
        interactor_config = MLPSAEnvInteractorConfig()
    elif wm.cfg.env.variant == "mlp_ma_env":
        train_env_config = get_mlp_ma_env_config(wm)
        eval_env_config = get_mlp_ma_env_config(wm)
        algorithm_config = get_algorithm_config(wm)
        interactor_config = MLPMAEnvInteractorConfig()
    elif wm.cfg.env.variant == "transformer_ma_env":
        train_env_config = get_transformer_ma_env_config(wm)
        eval_env_config = get_transformer_ma_env_config(wm)
        algorithm_config = get_algorithm_config(wm)
        interactor_config = TransformerMAEnvInteractorConfig()
    else:
        raise NotImplementedError

    memory_config = ReplayBufferConfig(
        mem_size=wm.cfg.replay_buffer.mem_size,
        mode=wm.cfg.replay_buffer.mode,
        device=str(wm.device),
        use_dict_wrapper=wm.cfg.replay_buffer.use_dict_wrapper,
        store_on_device=wm.cfg.replay_buffer.store_on_device,
        random_rollover=wm.cfg.replay_buffer.random_rollover,
    )

    # perform a run
    if wm.cfg.runner.variant == "sync":
        run_synchronous(
            wm=wm,
            train_env_config=train_env_config,
            eval_env_config=eval_env_config,
            algorithm_config=algorithm_config,
            memory_config=memory_config,
            interactor_config=interactor_config,
            settings=SynchronousRunnerSettings(
                max_transitions=wm.cfg.runner.max_transitions,
                transitions_per_epoch=wm.cfg.runner.transitions_per_epoch,
                transitions_num_exploration=wm.cfg.runner.transitions_num_exploration,
                transitions_min_for_train=wm.cfg.runner.transitions_min_for_train,
                eval_num_episodes=wm.cfg.runner.eval_num_episodes,
                eval_transitions_frequency=wm.cfg.runner.eval_transitions_frequency,
            ),
        )
    elif wm.cfg.runner.variant == "async":
        run_asynchronous(
            wm=wm,
            train_env_config=train_env_config,
            eval_env_config=eval_env_config,
            algorithm_config=algorithm_config,
            memory_config=memory_config,
            interactor_config=interactor_config,
            settings=AsynchronousRunnerSettings(
                num_parallel_rollouts=wm.cfg.runner.num_parallel_rollouts,
                queue_scale_ratio=wm.cfg.runner.queue_scale_ratio,
                max_transitions=wm.cfg.runner.max_transitions,
                transitions_per_epoch=wm.cfg.runner.transitions_per_epoch,
                transitions_num_exploration=wm.cfg.runner.transitions_num_exploration,
                transitions_min_for_train=wm.cfg.runner.transitions_min_for_train,
                eval_num_episodes=wm.cfg.runner.eval_num_episodes,
                eval_transitions_frequency=wm.cfg.runner.eval_transitions_frequency,
            ),
        )


def display(wm: Wingman) -> None:
    # TODO: this is kind of hacky now, try to streamline it
    if wm.cfg.env.variant == "mlp_sa_env":
        env = get_mlp_sa_env_config(wm).instantiate()
        alg = get_algorithm_config(wm).instantiate()
        has_weights, _, ckpt_dir = wm.get_weight_files()
        if has_weights:
            alg.load(ckpt_dir / "weights.pth")
        mlp_sa_env_display(env=env, actor=alg.actor)
    elif wm.cfg.env.variant == "mlp_ma_env":
        env = get_mlp_ma_env_config(wm).instantiate()
        alg = get_algorithm_config(wm).instantiate()
        has_weights, _, ckpt_dir = wm.get_weight_files()
        if has_weights:
            alg.load(ckpt_dir / "weights.pth")
        mlp_ma_env_display(env=env, actor=alg.actor)
    elif wm.cfg.env.variant == "transformer_ma_env":
        env = get_transformer_ma_env_config(wm).instantiate()
        alg = get_algorithm_config(wm).instantiate()
        has_weights, _, ckpt_dir = wm.get_weight_files()
        if has_weights:
            alg.load(ckpt_dir / "weights.pth")
        transformer_ma_env_display(env=env, actor=alg.actor)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)

    # fmt: off
    # config_yaml = Path(__file__).parent / "configs/quadx_waypoints_config.yaml"
    # config_yaml = (Path(__file__).parent / "configs/dual_dogfight_transformer_config.yaml")
    config_yaml = (Path(__file__).parent / "configs/async_dual_dogfight_transformer_config.yaml")
    # config_yaml = (Path(__file__).parent / "configs/quadx_ball_in_cup_config.yaml")
    # config_yaml = (Path(__file__).parent / "configs/cheetah_run_config.yaml")
    # config_yaml = (Path(__file__).parent / "configs/async_cheetah_run_config.yaml")
    # config_yaml = (Path(__file__).parent / "configs/quadx_waypoints_config.yaml")
    # fmt: on

    wm = Wingman(config_yaml=config_yaml)

    if wm.cfg.mode.train:
        train(wm)
    elif wm.cfg.mode.display:
        display(wm)
    else:
        print("So this is life now.")
