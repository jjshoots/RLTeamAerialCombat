import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

from wingman import Wingman

from dogfighter.runners.asynchronous.base import (AsynchronousRunnerSettings,
                                                  CollectionResult,
                                                  EvaluationResult,
                                                  TaskIOConfig, WorkerTaskType)
from dogfighter.runners.base import ConfigStack
from dogfighter.runners.utils import AtomicFileWriter


def _submit_task(
    mode: WorkerTaskType,
    task_io_config: TaskIOConfig,
    configs: ConfigStack,
) -> TaskIOConfig:
    task_config = configs.model_dump()
    task_config["runner_settings"]["mode"] = "worker"
    task_config["runner_settings"]["worker"]["task_io"]["actor_weights_path"] = (
        task_io_config.actor_weights_path
    )
    task_config["runner_settings"]["worker"]["task_io"]["result_output_path"] = (
        task_io_config.result_output_path
    )
    if mode == WorkerTaskType.COLLECT:
        task_config["runner_settings"]["worker"]["task"] = "collect"
    elif mode == WorkerTaskType.EVAL:
        task_config["runner_settings"]["worker"]["task"] = "eval"
    else:
        raise NotImplementedError

    # dump the file to disk and run the command
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(json.dumps(task_config).encode("utf-8"))
        f.flush()

        command = []
        command.append(f"{sys.executable}")
        command.append(str(Path(__file__).parent / "runner.py"))
        command.append("--task_file")
        command.append(f.name)

        # Run the command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    # print error if subprocess errors out
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        print(f"Error: {result.stdout}")

    return task_io_config


def run_train(
    wm: Wingman,
    configs: ConfigStack,
) -> None:
    algorithm_config = configs.algorithm_config
    memory_config = configs.memory_config
    settings = configs.runner_settings
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
    futures: dict[Future, WorkerTaskType] = {}
    with ProcessPoolExecutor(
        max_workers=settings.num_parallel_workers
    ) as exe, tempfile.TemporaryDirectory() as tmp_dir:
        actor_weights_path = ""

        while memory.count <= settings.transitions_max:
            """TASK ASSIGNMENT"""
            # conditionally add an eval task, this goes on before the collect tasks
            if memory.count >= next_eval_step:
                futures[
                    exe.submit(
                        _submit_task,
                        mode=WorkerTaskType.EVAL,
                        task_io_config=TaskIOConfig(
                            actor_weights_path=actor_weights_path,
                            result_output_path=f"{tmp_dir}/{uuid.uuid4()}_result.json",
                        ),
                        configs=configs,
                    )
                ] = WorkerTaskType.EVAL
                next_eval_step = (
                    int(memory.count / settings.transitions_eval_frequency) + 1
                ) * settings.transitions_eval_frequency

            # add as many collect tasks as needed
            while len(futures) < settings.max_tasks_in_queue:
                futures[
                    exe.submit(
                        _submit_task,
                        mode=WorkerTaskType.COLLECT,
                        task_io_config=TaskIOConfig(
                            actor_weights_path=actor_weights_path,
                            result_output_path=f"{tmp_dir}/{uuid.uuid4()}_result.json",
                        ),
                        configs=configs,
                    )
                ] = WorkerTaskType.COLLECT

            """RESULTS COLLECTION"""
            # check all futures for done tasks
            for future in list(futures.keys()):
                # skip this future if it's not done
                if not future.done():
                    continue

                # collect memory from workers
                if futures[future] == WorkerTaskType.COLLECT:
                    # load the results, merge the memory, cleanup
                    result_path = future.result().result_output_path
                    with open(result_path, "r") as f:
                        collection_result = CollectionResult(**json.load(f))
                    with open(collection_result.memory_path, "r+b") as f:
                        memory.merge(type(memory).load(f))
                    os.remove(collection_result.memory_path)
                    os.remove(result_path)

                    # update the log
                    wm.log.update(
                        {f"collect/{k}": v for k, v in collection_result.info.items()}
                    )

                # collect eval score from workers
                elif futures[future] == WorkerTaskType.EVAL:
                    # load the results, cleanup
                    result_path = future.result().result_output_path
                    with open(result_path, "r") as f:
                        evaluation_result = EvaluationResult(**json.load(f))
                    os.remove(result_path)

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

            # update the actor weights for workers, use a rename to prevent race
            with AtomicFileWriter(f"{tmp_dir}/actor_weights.pth") as f:
                algorithm.actor.save(f)

            # save weights
            to_update, _, ckpt_dir = wm.checkpoint(loss=-eval_score, step=memory.count)
            if to_update:
                algorithm.save(ckpt_dir / "weights.pth")
