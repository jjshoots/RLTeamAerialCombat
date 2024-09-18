import json
import os
import tempfile

import torch

from dogfighter.runners.asynchronous.base import (
    AsynchronousRunnerSettings,
    CollectionResult,
    EvaluationResult,
)
from dogfighter.runners.base import ConfigStack
from dogfighter.runners.utils import AtomicFileWriter


def run_collection(
    configs: ConfigStack,
    actor_weights_path: str,
    result_output_path: str,
) -> None:
    """Collection worker.

    The outputs of this function are:
    1. CollectionResult
        - The location of the collected replay buffer
        - Collection info
    2. The path to the JSON above is saved in `settings.runner.worker.result_output_path`, specified from the input.
    """
    # splice out configs
    train_env_config = configs.train_env_config
    algorithm_config = configs.algorithm_config
    memory_config = configs.memory_config
    interactor_config = configs.interactor_config
    settings = configs.runner_settings
    assert isinstance(settings, AsynchronousRunnerSettings)

    # instantiate things
    env = train_env_config.instantiate()
    actor = algorithm_config.actor_config.instantiate()
    collection_fn = interactor_config.get_collection_fn()
    memory = memory_config.model_copy(
        update={"mem_size": int(settings.collect.buffer_size)}
    ).instantiate()

    # load the weights file
    if has_weights := os.path.exists(actor_weights_path):
        actor.load(actor_weights_path)

    # send to gpu and compile
    actor.to(algorithm_config.device)
    torch.compile(actor)

    # run a collect task
    memory, info = collection_fn(
        actor=actor,
        env=env,
        memory=memory,
        use_random_actions=not has_weights,
        num_transitions=settings.collect.min_transitions,
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
    assert result_output_path is not None
    with AtomicFileWriter(result_output_path) as f:
        with open(f, "w") as fw:
            json.dump(result.model_dump(), fw)


def run_evaluation(
    configs: ConfigStack,
    actor_weights_path: str,
    result_output_path: str,
) -> None:
    """Evaluation worker.

    To run this, need to provide:
    1. actor_weight_path (a filepath of where the weights for the actor are) [Optional]
    2. result_output_path (a filepath of where the results json should be placed)

    The outputs of this function are:
    1. EvaluationResult
        - Evaluation score
        - Collection info
    2. The path to the JSON above is saved in `settings.runner.worker.result_output_path`, specified from the input.
    """
    # splice out configs
    eval_env_config = configs.eval_env_config
    algorithm_config = configs.algorithm_config
    interactor_config = configs.interactor_config
    settings = configs.runner_settings
    assert isinstance(settings, AsynchronousRunnerSettings)

    # instantiate things
    env = eval_env_config.instantiate()
    actor = algorithm_config.actor_config.instantiate()
    evaluation_fn = interactor_config.get_evaluation_fn()

    # load the weights file
    if os.path.exists(actor_weights_path) is not None:
        actor.load(actor_weights_path)

    # send to gpu and compile
    actor.to(algorithm_config.device)
    torch.compile(actor)

    # run an eval task
    eval_score, info = evaluation_fn(
        actor=actor,
        env=env,
        num_episodes=settings.evaluate.num_episodes,
    )

    # form the results
    result = EvaluationResult(
        score=eval_score,
        info=info,
    )

    # dump the pointer to disk
    assert result_output_path is not None
    with AtomicFileWriter(result_output_path) as f:
        with open(f, "w") as fw:
            json.dump(result.model_dump(), fw)
