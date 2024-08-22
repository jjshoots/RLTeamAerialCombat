from __future__ import annotations

import math

from pydantic import BaseModel
from wingman import Wingman
import time

from dogfighter.bases.base_algorithm import AlgorithmConfig
from dogfighter.bases.base_env_creators import MAEnvConfig, SAVecEnvConfig
from dogfighter.bases.base_env_interactors import (CollectFunctionProtocol,
                                                   EvaluationFunctionProtocol)
from dogfighter.bases.base_replay_buffer import ReplayBufferConfig


class SynchronousRunnerSettings(BaseModel):
    """SynchronousRunnerSettings."""

    max_transitions: int
    transitions_per_epoch: int
    transitions_num_exploration: int

    train_min_transitions: int

    eval_num_episodes: int
    eval_transitions_frequency: int


def run_synchronous(
    wm: Wingman,
    train_env_config: SAVecEnvConfig | MAEnvConfig,
    eval_env_config: SAVecEnvConfig | MAEnvConfig,
    algorithm_config: AlgorithmConfig,
    memory_config: ReplayBufferConfig,
    collect_fn: CollectFunctionProtocol,
    evaluation_fn: EvaluationFunctionProtocol,
    settings: SynchronousRunnerSettings,
) -> None:
    """A synchronous runner to perform train and evaluations in step.

    Args:
        wm (Wingman): wm
        train_env_config (SAVecEnvConfig | MAEnvConfig): train_env_config
        eval_env_config (SAVecEnvConfig | MAEnvConfig): eval_env_config
        algorithm_config (AlgorithmConfig): algorithm_config
        memory_config (ReplayBufferConfig): memory_config
        collect_fn (CollectFunctionProtocol): collect_fn
        evaluation_fn (EvaluationFunctionProtocol): evaluation_fn
        settings (SynchronousRunnerSettings): settings

    Returns:
        None:
    """
    # instantiate everything
    train_env = train_env_config.instantiate()
    eval_env = eval_env_config.instantiate()
    algorithm = algorithm_config.instantiate()
    memory = memory_config.instantiate()

    # get latest weight files
    has_weights, _, ckpt_dir = wm.get_weight_files()
    if has_weights:
        algorithm.load(ckpt_dir / "weights.pth")

    # logging metrics
    num_epochs = 0
    eval_score = -math.inf
    max_eval_score = -math.inf
    next_eval_step = 0
    loop_start_time = time.time()

    # start the main training loop
    while memory.count <= settings.max_transitions:
        print("\n\n")
        print(
            f"New epoch @ {memory.count} / {settings.max_transitions} total transitions."
        )
        num_epochs += 1

        """POLICY ROLLOUT"""
        memory, info = collect_fn(
            actor=algorithm.actor,
            env=train_env,
            memory=memory,
            use_random_actions=memory.count < settings.transitions_num_exploration,
            num_transitions=settings.transitions_per_epoch,
        )
        wm.log.update({f"collect/{k}": v for k, v in info.items()})

        # don't proceed with training until we have a minimum number of transitions
        if memory.count < settings.train_min_transitions:
            print(
                "Haven't reached minimum number of transitions "
                f"({memory.count} / {settings.train_min_transitions}) "
                "required before training, continuing with sampling..."
            )
            continue

        """TRAINING RUN"""
        print(
            f"Training epoch {num_epochs}, "
            f"Replay Buffer Capacity {memory.count} / {memory.mem_size}"
        )
        info = algorithm.update(memory=memory)
        wm.log.update({f"train/{k}": v for k, v in info.items()})

        """EVALUATE POLICY"""
        if memory.count >= next_eval_step:
            eval_score, info = evaluation_fn(
                actor=algorithm.actor,
                env=eval_env,
                num_episodes=settings.eval_num_episodes,
            )
            wm.log.update({f"eval/{k}": v for k, v in info.items()})
            max_eval_score = max(max_eval_score, eval_score)
            next_eval_step = (
                int(memory.count / settings.eval_transitions_frequency) + 1
            ) * settings.eval_transitions_frequency
        wm.log["eval/score"] = eval_score
        wm.log["eval/max_score"] = max_eval_score

        """LOGGING"""
        # record looptimes
        looptime = time.time() - loop_start_time
        loop_start_time = time.time()
        print(
            "ETA to completion: "
            f"{
                (
                    (settings.max_transitions - memory.count)
                    * (settings.transitions_per_epoch / looptime)
                ):.0f
            } seconds..."
        )

        # collect some statistics
        wm.log["runner/epoch"] = 0
        wm.log["runner/memory_size"] = memory.__len__()
        wm.log["runner/num_transitions"] = memory.count
        wm.log["runner/looptime"] = looptime

        # save weights
        to_update, _, ckpt_dir = wm.checkpoint(loss=-eval_score, step=memory.count)
        if to_update:
            algorithm.save(ckpt_dir / "weights.pth")
