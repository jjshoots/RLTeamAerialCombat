import json
import math
import os
import tempfile
import time
import uuid
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

from wingman import Wingman

from dogfighter.algorithms.base import AlgorithmConfig
from dogfighter.replay_buffers.replay_buffer import ReplayBufferConfig
from dogfighter.runners.asynchronous_runner.base import (
    AsynchronousRunnerSettings, CollectionResult, EvaluationResult, WorkerMode)
from dogfighter.runners.synchronous_runner import SynchronousRunnerSettings


def submit_task(
    mode: WorkerMode, actor_weights_path: str, result_output_path: str
) -> str:
    # TODO: submit the task here

    while not Path(result_output_path).exists():
        time.sleep(1)

    return result_output_path


def run_train(
    wm: Wingman,
    algorithm_config: AlgorithmConfig,
    memory_config: ReplayBufferConfig,
    settings: SynchronousRunnerSettings | AsynchronousRunnerSettings,
) -> None:
    # we only want trainer settings
    assert isinstance(settings, AsynchronousRunnerSettings)

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

    # run things using executor
    futures: dict[Future, WorkerMode] = {}
    with ProcessPoolExecutor(
        max_workers=settings.num_parallel_workers
    ) as exe, tempfile.TemporaryDirectory() as tmp_dir:
        while memory.count <= settings.transitions_max:
            """TASK ASSIGNMENT"""
            # conditionally add an eval task, this goes on before the collect tasks
            if memory.count >= next_eval_step:
                # form the actor_weights_path and result_output_path
                _task_uuid = uuid.uuid4()
                result_output_path = f"{tmp_dir}/{_task_uuid}_collect_results.pth"
                if memory.count >= settings.transitions_num_exploration:
                    actor_weights_path = f"{tmp_dir}/{_task_uuid}_actor_weights.pth"
                    algorithm.save(actor_weights_path)
                else:
                    actor_weights_path = ""

                # assign the task
                futures[
                    exe.submit(
                        submit_task,
                        mode=WorkerMode.EVAL,
                        actor_weights_path=actor_weights_path,
                        result_output_path=result_output_path,
                    )
                ] = WorkerMode.EVAL
                next_eval_step = (
                    int(memory.count / settings.transitions_eval_frequency) + 1
                ) * settings.transitions_eval_frequency

            # add as many collect tasks as needed
            while len(futures) < settings.max_tasks_in_queue:
                # form the actor_weights_path and result_output_path
                _task_uuid = uuid.uuid4()
                result_output_path = f"{tmp_dir}/{_task_uuid}_collect_results.pth"
                if memory.count >= settings.transitions_num_exploration:
                    actor_weights_path = f"{tmp_dir}/{_task_uuid}_actor_weights.pth"
                    algorithm.actor.save(actor_weights_path)
                else:
                    actor_weights_path = ""

                # assign the task
                futures[
                    exe.submit(
                        submit_task,
                        mode=WorkerMode.COLLECT,
                        actor_weights_path=actor_weights_path,
                        result_output_path=result_output_path,
                    )
                ] = WorkerMode.COLLECT

            """RESULTS COLLECTION"""
            # check all futures for done tasks
            for future in list(futures.keys()):
                # skip this future if it's not done
                if not future.done():
                    continue

                # collect memory from workers
                if futures[future] == WorkerMode.COLLECT:
                    # load the results
                    with open(future.result(), "r") as f:
                        collection_result = CollectionResult(**json.load(f))

                    # merge the memory and cleanup
                    with open(collection_result.memory_path, "r+b") as f:
                        memory.merge(type(memory).load(f))
                    os.remove(collection_result.memory_path)

                    # update the log
                    wm.log.update(
                        {f"collect/{k}": v for k, v in collection_result.info.items()}
                    )

                # collect eval score from workers
                elif futures[future] == WorkerMode.EVAL:
                    # load the results
                    with open(future.result(), "r") as f:
                        evaluation_result = EvaluationResult(**json.load(f))

                    # splice out the info and add to log
                    eval_score, info = evaluation_result.score, evaluation_result.info
                    max_eval_score = max(max_eval_score, eval_score)
                    wm.log["eval/score"] = eval_score
                    wm.log["eval/max_score"] = max_eval_score
                    wm.log.update({f"eval/{k}": v for k, v in info.items()})
                else:
                    raise NotImplementedError

                # clear the future from the list
                del futures[future]

            """TRAINING RUN"""
            # don't proceed with training until we have a minimum number of transitions
            if memory.count < settings.transitions_min_for_train:
                time.sleep(5)
                print(
                    "Haven't reached minimum number of transitions "
                    f"({memory.count} / {settings.transitions_min_for_train}) "
                    "required before training, continuing with sampling..."
                )
                continue

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
