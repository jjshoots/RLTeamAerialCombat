from __future__ import annotations

from signal import SIGINT, signal

import torch
from wingman import Wingman
from wingman.utils import shutdown_handler

from ma_env_interaction_utils import ma_env_collect_to_memory, ma_env_evaluate
from setup_utils import (setup_algorithm, setup_ma_environment,
                         setup_replay_buffer)


def train(wm: Wingman) -> None:
    # pull the config out of wingman
    cfg = wm.cfg

    # setup envs, alg, replay buffer
    train_env = setup_ma_environment(wm)
    eval_env = setup_ma_environment(wm)
    alg = setup_algorithm(wm)
    memory = setup_replay_buffer(wm)

    # logging metrics
    wm.log["epoch"] = 0
    next_eval_step = 0

    # start the main training loop
    while memory.count <= cfg.total_steps:
        print("\n\n")
        print(f"New epoch @ {memory.count} / {cfg.total_steps} total transitions.")
        wm.log["epoch"] += 1

        """POLICY ROLLOUT"""
        memory, info = ma_env_collect_to_memory(
            actor=alg.actor,
            ma_env=train_env,
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
            info = ma_env_evaluate(
                actor=alg.actor,
                ma_env=eval_env,
                num_episodes=cfg.eval_num_episodes,
            )
            wm.log.update(info)
            wm.log["performance"] = (
                info["mean_episode_interactions"] + info["mean_hits_per_agent"] * 10.0
            ) / (info["num_collisions"] + info["num_out_of_bounds"] + 1)
            next_eval_step = (
                int(memory.count / cfg.eval_steps_ratio) + 1
            ) * cfg.eval_steps_ratio

        """WANDB"""
        # save weights
        to_update, model_file, _ = wm.checkpoint(
            loss=-float(wm.log["performance"]), step=wm.log["num_transitions"]
        )
        if to_update:
            torch.save(alg.state_dict(), model_file)


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    wm = Wingman(config_yaml="./configs/dual_dogfight_config.yaml")

    if wm.cfg.train:
        train(wm)
    elif wm.cfg.display:
        info = ma_env_evaluate(
            ma_env=setup_ma_environment(wm),
            actor=setup_algorithm(wm).actor,
            num_episodes=1,
        )
    else:
        print("So this is life now.")
