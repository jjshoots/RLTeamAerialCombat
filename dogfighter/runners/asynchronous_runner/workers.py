import json
import os
import tempfile
import time

from wingman import Wingman

from dogfighter.runners.asynchronous_runner.base import (
    AsynchronousRunnerSettings, CollectionResult, EvaluationResult, TaskConfig)
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

    Things to override for through CLI from trainer:
    1. model.id
    2. runner.mode
    3. runner.worker.task
    """
    # load all the configs for this run
    (
        train_env_config,
        eval_env_config,
        interactor_config,
        algorithm_config,
        memory_config,
        settings,
    ) = get_all_configs(wm)

    # we only want to use worker settings
    assert isinstance(settings, AsynchronousRunnerSettings)
    settings = settings.worker

    # instantiate things
    env = train_env_config.instantiate()
    actor = algorithm_config.actor_config.instantiate()
    collection_fn = interactor_config.get_collection_fn()
    memory = memory_config.model_copy(
        update={"mem_size": int(settings.collect_num_transitions * 1.2)}
    ).instantiate()

    # load the weights file and clean up
    if settings.actor_weights_path:
        actor.load(settings.actor_weights_path)
        os.remove(settings.actor_weights_path)

    # run a collect task
    memory, info = collection_fn(
        actor=actor.actor,
        env=env,
        memory=memory,
        use_random_actions=bool(settings.actor_weights_path),
        num_transitions=settings.collect_num_transitions,
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

    temp_path = f"{settings.result_output_path}.temp.json"
    with open(temp_path, "w") as f:
        json.dump(result, f)
        time.sleep(1)
    os.rename(temp_path, settings.result_output_path)


def run_evaluation(wm: Wingman) -> None:
    """Evaluation worker.

    To run this, need to provide:
    1. In a JSON:
        - actor_weight_path (a filepath of where the weights for the actor are) [Optional]
        - results_path (a filepath of where the results json should be placed)
    2. The path to the JSON above as `wm.cfg.runner.worker.task_config_path`, through the CLI

    The outputs of this function are:
    1. EvaluationResult
        - Evaluation score
        - Collection info
    2. The path to the JSON above is saved in `task_config.results_path`, specified from the input.

    Things to override for through CLI from trainer:
    1. model.id
    2. runner.mode
    3. runner.worker.task
    """
    (
        train_env_config,
        eval_env_config,
        interactor_config,
        algorithm_config,
        memory_config,
        settings,
    ) = get_all_configs(wm)

    # we only want to use worker settings
    assert isinstance(settings, AsynchronousRunnerSettings)
    settings = settings.worker

    # instantiate things
    env = eval_env_config.instantiate()
    actor = algorithm_config.actor_config.instantiate()
    evaluation_fn = interactor_config.get_evaluation_fn()

    # load the weights file and clean up
    if settings.actor_weights_path:
        actor.load(settings.actor_weights_path)
        os.remove(settings.actor_weights_path)

    # run an eval task
    eval_score, info = evaluation_fn(
        actor=actor.actor,
        env=env,
        num_episodes=settings.eval_num_episodes,
    )

    # form the results
    result = EvaluationResult(
        score=eval_score,
        info=info,
    )

    # dump the pointer to disk, we do a write, then rename
    # this way, the file can't be read while it's being written
    temp_path = f"{settings.result_output_path}.temp.json"
    with open(temp_path, "w") as f:
        json.dump(result, f)
        time.sleep(1)
    os.rename(temp_path, settings.result_output_path)
