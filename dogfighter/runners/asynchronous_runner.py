from __future__ import annotations

import math
import os
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel
from wingman import Wingman

from dogfighter.bases.base_algorithm import AlgorithmConfig
from dogfighter.bases.base_env_creators import MAEnvConfig, SAVecEnvConfig
from dogfighter.bases.base_env_interactors import (CollectFunctionProtocol,
                                                   EvaluationFunctionProtocol)
from dogfighter.bases.base_replay_buffer import ReplayBufferConfig
from wingman.replay_buffer import ReplayBuffer

import random
from concurrent.futures import ThreadPoolExecutor, Future
import tempfile

class AsynchronousRunnerSettings(BaseModel):
    """SynchronousRunnerSettings."""

    num_parallel_rollouts: int
    queue_scale_ratio: int = 8

    transitions_per_epoch: int
    transitions_num_exploration: int
    transitions_min_for_train: int
    max_transitions: int

    eval_num_episodes: int
    eval_transitions_frequency: int


class _WorkerMode(Enum):
    COLLECT = 1
    EVAL = 2


def _collect_worker(
    weight_path: None | str,
    env_config: SAVecEnvConfig | MAEnvConfig,
    algorithm_config: AlgorithmConfig,
    memory_config: ReplayBufferConfig,
    collect_fn: CollectFunctionProtocol,
    settings: AsynchronousRunnerSettings,
) -> tuple[ReplayBuffer, dict[str, Any]]:
    # instantiate things
    env = env_config.instantiate()
    algorithm = algorithm_config.instantiate()
    memory = memory_config.instantiate()

    # get latest weight files if it exists, and clean up
    if weight_path:
        algorithm.load(weight_path)
        os.remove(weight_path)

    # run a collect task
    memory, info = collect_fn(
        actor=algorithm.actor,
        env=env,
        memory=memory,
        use_random_actions=weight_path is not None,
        num_transitions=settings.transitions_per_epoch,
    )

    return memory, info


def _eval_worker(
    weight_path: None | str,
    env_config: SAVecEnvConfig | MAEnvConfig,
    algorithm_config: AlgorithmConfig,
    evaluation_fn: EvaluationFunctionProtocol,
    settings: AsynchronousRunnerSettings,
) -> tuple[ReplayBuffer, dict[str, Any]]:
    # instantiate things
    env = env_config.instantiate()
    algorithm = algorithm_config.instantiate()

    # get latest weight files if it exists, and clean up
    if weight_path:
        algorithm.load(weight_path)
        os.remove(weight_path)

    # run an eval task
    eval_score, info = evaluation_fn(
        actor=algorithm.actor,
        env=env,
        num_episodes=settings.eval_num_episodes,
    )

    return eval_score, info


def run_asynchronous(
    wm: Wingman,
    train_env_config: SAVecEnvConfig | MAEnvConfig,
    eval_env_config: SAVecEnvConfig | MAEnvConfig,
    algorithm_config: AlgorithmConfig,
    memory_config: ReplayBufferConfig,
    collect_fn: CollectFunctionProtocol,
    evaluation_fn: EvaluationFunctionProtocol,
    settings: AsynchronousRunnerSettings,
) -> None:
    # instantiate everything
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

    # run things using executor
    with ThreadPoolExecutor(max_workers=settings.num_parallel_rollouts) as exe:
        futures: dict[Future, _WorkerMode] = {}

        while memory.count <= settings.max_transitions:
            """TASK ASSIGNMENT"""
            # conditionally add an eval task, this goes on before the collect tasks
            if memory.count >= next_eval_step:
                # if we can start using weights, we need to assign weights paths
                if memory.count >= settings.transitions_num_exploration:
                    fd, weight_path = tempfile.mkstemp(suffix=".pth")
                    os.close(fd)
                    algorithm.save(weight_path)
                else:
                    weight_path = None

                # assign the task
                futures[
                    exe.submit(
                        _eval_worker,
                        weight_path=weight_path,
                        env_config=train_env_config,
                        algorithm_config=algorithm_config,
                        evaluation_fn=evaluation_fn,
                        settings=settings,
                    )
                ] = _WorkerMode.EVAL
                next_eval_step = (
                    int(memory.count / settings.eval_transitions_frequency) + 1
                ) * settings.eval_transitions_frequency

            # add as many collect tasks as needed
            while len(futures) < settings.num_parallel_rollouts * settings.queue_scale_ratio:
                # if we can start using weights, we need to assign weights paths
                if memory.count >= settings.transitions_num_exploration:
                    fd, weight_path = tempfile.mkstemp(suffix=".pth")
                    os.close(fd)
                    algorithm.save(weight_path)
                else:
                    weight_path = None

                futures[
                    exe.submit(
                        _collect_worker,
                        weight_path=weight_path,
                        env_config=eval_env_config,
                        algorithm_config=algorithm_config,
                        memory_config=memory_config,
                        collect_fn=collect_fn,
                        settings=settings,
                    )
                ] = _WorkerMode.COLLECT

            """RESULTS COLLECTION"""
            # check all futures for done tasks
            done_futures = []
            for future in list(futures.keys()):
                print(len(futures))
                # skip this future if it's not done
                if not future.done():
                    continue

                # collect memory from workers
                if futures[future] == _WorkerMode.COLLECT:
                    collect_memory, info = future.result()
                    wm.log.update({f"collect/{k}": v for k, v in info.items()})
                    memory.merge(collect_memory)
                # collect eval score from workers
                elif futures[future] == _WorkerMode.EVAL:
                    eval_score, info = future.result()
                    max_eval_score = max(max_eval_score, eval_score)
                    wm.log.update({f"eval/{k}": v for k, v in info.items()})
                    wm.log["eval/score"] = eval_score
                    wm.log["eval/max_score"] = max_eval_score
                else:
                    raise NotImplementedError

                # clear the future from the list
                del futures[future]

            """TRAINING RUN"""
            print(
                f"Training epoch {num_epochs}, "
                f"Replay Buffer Capacity {memory.count} / {memory.mem_size}"
            )
            info = algorithm.update(memory=memory)
            wm.log.update({f"train/{k}": v for k, v in info.items()})

            # save weights
            to_update, _, ckpt_dir = wm.checkpoint(loss=-eval_score, step=memory.count)
            if to_update:
                algorithm.save(ckpt_dir / "weights.pth")
