import math
import os
import time
from collections import defaultdict

from wingman import Wingman

from dogfighter.runners.asynchronous.base import (
    AsynchronousRunnerSettings,
    CollectionResult,
    EvaluationResult,
)
from dogfighter.runners.asynchronous.task_dispatcher import TaskDispatcher
from dogfighter.runners.base import ConfigStack
from dogfighter.runners.utils import AtomicFileWriter


def run_train(
    wm: Wingman,
    configs: ConfigStack,
) -> None:
    algorithm_config = configs.algorithm_config
    memory_config = configs.memory_config
    assert isinstance(configs.runner_settings, AsynchronousRunnerSettings)
    settings = configs.runner_settings.trainer

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
    train_start_time = time.time()

    # start main loop and dispatcher
    task_dispatcher = TaskDispatcher(config_stack=configs)
    while memory.count <= settings.transitions_max:
        loop_start_time = time.time()
        num_epochs += 1

        """UPDATE WORKER WEIGHTS"""
        if memory.count > settings.transitions_num_exploration:
            with AtomicFileWriter(task_dispatcher.actor_weights_path) as f:
                algorithm.actor.save(f)

        """COLLECT RESULTS"""
        for result in task_dispatcher.completed_tasks:
            collect_infos = defaultdict(list)
            eval_infos = defaultdict(list)
            eval_scores = []

            # collect results
            if isinstance(result, CollectionResult):
                with open(result.memory_path, "r+b") as f:
                    memory.merge(type(memory).load(f))
                os.remove(result.memory_path)
                for key, value in result.info.items():
                    collect_infos[key].append(value)
            elif isinstance(result, EvaluationResult):
                eval_scores.append(result.score)
                for key, value in result.info.items():
                    eval_infos[key].append(value)
            else:
                raise NotImplementedError

            # aggregate results
            if eval_scores:
                wm.log["eval/score"] = eval_score = sum(eval_scores) / len(eval_scores)
                wm.log["eval/max_score"] = max_eval_score = max(
                    max_eval_score, eval_score
                )
            wm.log.update(
                {f"collect/{k}": (sum(v) / len(v)) for k, v in collect_infos.items()}
            )
            wm.log.update(
                {f"eval/{k}": (sum(v) / len(v)) for k, v in eval_infos.items()}
            )

        """MODEL TRAINING"""
        # don't proceed with training until we have a minimum number of transitions
        if memory.count < settings.transitions_min_for_train:
            time.sleep(1)
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

        """SEND EVAL"""
        task_dispatcher.queue_eval()

        """LOGGING"""
        # collect some statistics
        wm.log["runner/epoch"] = num_epochs
        wm.log["runner/memory_size"] = memory.__len__()
        wm.log["runner/num_transitions"] = memory.count
        wm.log["runner/looptime"] = time.time() - loop_start_time
        wm.log["runner/eta_completion"] = (
            (time.time() - train_start_time) / memory.count * settings.transitions_max
        )

        # print things
        print(f"ETA to completion: {wm.log['runner/eta_completion']:.0f} seconds...")

        to_update, _, ckpt_dir = wm.checkpoint(loss=-eval_score, step=num_epochs)
        if to_update:
            algorithm.save(ckpt_dir / "weights.pth")
