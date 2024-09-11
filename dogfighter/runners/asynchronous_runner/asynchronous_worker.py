import fcntl
import json
import os
import tempfile
import time
from typing import Any

from wingman import Wingman

from dogfighter.runners.asynchronous_runner.base import (
    AsynchronousRunnerSettings, CollectionResult, TaskConfig)
from setup_configs import get_all_configs


def run_collection(wm: Wingman) -> None:
    """Collection worker.

    To run this, need to provide:
    1. In a JSON:
        - actor_weight_path (a filepath of where the weights for the actor are) [Optional]
        - results_path (a filepath of where the results json should be placed)
    2. The path to the JSON above as `wm.cfg.runner.worker.task_config_path`, through the CLI

    The outputs of this function are:
    1. CollectionResult
        - The location of the collected replay buffer
        - Collection info
    2. The path to the JSON above is saved in `task_config.results_path`, specified from the input.
    """
    # load all the configs for this run
    (
        train_env_config,
        eval_env_config,
        interactor_config,
        algorithm_config,
        memory_config,
        runner_settings,
    ) = get_all_configs(wm)

    # we only want to use worker settings
    assert isinstance(runner_settings, AsynchronousRunnerSettings)
    runner_settings = runner_settings.worker

    # instantiate things
    env = train_env_config.instantiate()
    actor = algorithm_config.actor_config.instantiate()
    memory = memory_config.instantiate()
    collection_fn = interactor_config.get_collection_fn()

    # load the task config
    with open(runner_settings.task_config_path, "r") as f:
        task_config = TaskConfig.model_validate_json(json_data=json.load(f))
    os.remove(wm.cfg.runner.worker.task_config_path)

    # load the weights file and clean up
    if task_config.actor_weight_path:
        actor.load(task_config.actor_weight_path)
        os.remove(task_config.actor_weight_path)

    # run a collect task
    memory, info = collection_fn(
        actor=actor.actor,
        env=env,
        memory=memory,
        use_random_actions=task_config.actor_weight_path is not None,
        num_transitions=runner_settings.collect_num_transitions,
    )

    # dump the memory to disk
    fd, memory_path = tempfile.mkstemp(suffix=".zip")
    with open(memory_path, "w+b") as f:
        memory.dump(f)
    os.close(fd)

    # form the results
    result = CollectionResult(
        memory_path=memory_path,
        info=info,
    )

    # dump the pointer to disk
    with open(task_config.results_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(result, f)
        time.sleep(1)
        fcntl.flock(f, fcntl.LOCK_UN)


def run_evaluation(
    wm: Wingman,
    actor_weight_path: str,
) -> tuple[float, dict[str, Any]]:
    """run_evaluation.

    Args:
        wm (Wingman): wm
        actor_weight_path (str): actor_weight_path

    Returns:
        tuple[float, dict[str, Any]]:
    """
    (
        train_env_config,
        eval_env_config,
        interactor_config,
        algorithm_config,
        memory_config,
        runner_settings,
    ) = get_all_configs(wm)

    assert isinstance(runner_settings, AsynchronousRunnerSettings)
    # instantiate things
    env = eval_env_config.instantiate()
    actor = algorithm_config.actor_config.instantiate()
    evaluation_fn = interactor_config.get_evaluation_fn()

    # get latest weight files if it exists, and clean up
    if actor_weight_path:
        actor.load(actor_weight_path)
        os.remove(actor_weight_path)

    # run an eval task
    eval_score, info = evaluation_fn(
        actor=actor.actor,
        env=env,
        num_episodes=runner_settings.eval_num_episodes,
    )

    return eval_score, info
