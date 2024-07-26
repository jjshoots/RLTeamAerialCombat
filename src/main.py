from __future__ import annotations

import math
from signal import SIGINT, signal

import torch
from wingman import Wingman
from wingman.utils import shutdown_handler

from ma_env_interaction_utils import ma_env_collect_to_memory, ma_env_evaluate
from vec_env_interaction_utils import vec_env_render_gif, vec_env_collect_to_memory, vec_env_evaluate
from setup_utils import (setup_algorithm, setup_ma_environment,
                         setup_replay_buffer, setup_vector_environment)


def train(wm: Wingman) -> None:
    # pull the config out of wingman
    cfg = wm.cfg

    # setup envs, alg, replay buffer
    if cfg.env_type == "vec_env":
        train_env = setup_vector_environment(wm)
        eval_env = setup_vector_environment(wm)
        collect_function = vec_env_collect_to_memory
        evaluate_function = vec_env_evaluate
    elif cfg.env_type == "ma_env":
        train_env = setup_ma_environment(wm)
        eval_env = setup_ma_environment(wm)
        collect_function = ma_env_collect_to_memory
        evaluate_function = ma_env_evaluate
    else:
        raise ValueError(f"Expected only 'vec_env' and 'ma_env' for env_type, got '{cfg.env_type}'.")

    # setup algorithm and replay buffer
    alg = setup_algorithm(wm)
    memory = setup_replay_buffer(wm)

    # logging metrics
    wm.log["epoch"] = 0
    wm.log["max_eval_perf"] = -math.inf
    next_eval_step = 0

    # start the main training loop
    while memory.count <= cfg.total_steps:
        print("\n\n")
        print(f"New epoch @ {memory.count} / {cfg.total_steps} total transitions.")
        wm.log["epoch"] += 1

        """POLICY ROLLOUT"""
        memory, info = collect_function(
            actor=alg.actor,
            env=train_env,  # pyright: ignore[reportArgumentType]
            memory=memory,
            random_actions=memory.count < cfg.exploration_steps,
            num_transitions=cfg.env_transitions_per_epoch,
        )
        wm.log["buffer_size"] = memory.__len__()
        wm.log["num_transitions"] = memory.count
        wm.log.update(info)

        # don't proceed with training until we have a minimum number of transitions
        if memory.count < cfg.min_transitions_before_training:
            print(
                f"Haven't reached minimum number of transitions ({memory.count} / {cfg.min_transitions_before_training}) "
                "required before training, continuing with sampling..."
            )
            continue

        """TRAINING RUN"""
        print(
            f"Training epoch {wm.log['epoch']}, Replay Buffer Capacity {memory.count} / {memory.mem_size}"
        )
        info = alg.update(
            memory=memory,
            batch_size=cfg.batch_size,
            num_gradient_steps=cfg.model_updates_per_epoch,
        )
        wm.log.update(info)

        """EVALUATE POLICY"""
        if memory.count >= next_eval_step:
            info = evaluate_function(
                actor=alg.actor,
                env=eval_env,  # pyright: ignore[reportArgumentType]
                num_episodes=cfg.eval_num_episodes,
            )
            wm.log.update(info)
            wm.log["max_eval_perf"] = max(
                [float(wm.log["max_eval_perf"]), float(wm.log["eval_perf"])]
            )
            next_eval_step = (
                int(memory.count / cfg.eval_steps_ratio) + 1
            ) * cfg.eval_steps_ratio

        """WANDB"""
        # save weights
        to_update, model_file, _ = wm.checkpoint(
            loss=-float(wm.log["eval_perf"]), step=wm.log["num_transitions"]
        )
        if to_update:
            torch.save(alg.state_dict(), model_file)


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    # wm = Wingman(config_yaml="./configs/quad_dogfight_config.yaml")
    wm = Wingman(config_yaml="./configs/dual_dogfight_config.yaml")

    if wm.cfg.train:
        train(wm)
    elif wm.cfg.display:
        if wm.cfg.env_type == "ma_env":
            info = ma_env_evaluate(
                env=setup_ma_environment(wm),
                actor=setup_algorithm(wm).actor,
                num_episodes=1,
            )
        elif wm.cfg.env_type == "vec_env":
            wm.cfg.num_envs = 1 if wm.cfg.display else wm.cfg.num_envs
            wm.log["eval_perf"], wm.log["mean_episode_length"] = vec_env_evaluate(
                env=setup_vector_environment(wm),
                actor=setup_algorithm(wm).actor,
                num_episodes=wm.cfg.eval_num_episodes,
            )
        else:
            raise ValueError(f"Expected only 'vec_env' and 'ma_env' for env_type, got '{cfg.env_type}'.")
    elif wm.cfg.render:
        if wm.cfg.env_type == "vec_env":
            print(vec_env_render_gif(wm=wm))
    else:
        print("So this is life now.")
