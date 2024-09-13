import json
import os
import tempfile

from dogfighter.runners.asynchronous.base import (AsynchronousRunnerSettings,
                                                  CollectionResult,
                                                  EvaluationResult)
from dogfighter.runners.base import ConfigStack
from dogfighter.runners.utils import AtomicFileWriter


def run_collection(
    configs: ConfigStack,
) -> None:
    """Collection worker.

    To run this, need to provide:
    1. actor_weight_path (a filepath of where the weights for the actor are) [Optional]
    2. result_output_path (a filepath of where the results json should be placed)

    The outputs of this function are:
    1. CollectionResult
        - The location of the collected replay buffer
        - Collection info
    2. The path to the JSON above is saved in `settings.runner.worker.result_output_path`, specified from the input.

    Things to override for through CLI from trainer:
    1. model.id
    2. runner.mode
    3. runner.worker.task
    4. runner.worker.actor_weights_path
    5. runner.worker.result_output_path
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
    actor.to(algorithm_config.device)
    memory = memory_config.model_copy(
        update={"mem_size": int(settings.worker_settings.collect_num_transitions * 1.2)}
    ).instantiate()

    # load the weights file and clean up
    if settings.worker_settings.io.actor_weights_path:
        actor.load(settings.worker_settings.io.actor_weights_path)

    # run a collect task
    memory, info = collection_fn(
        actor=actor,
        env=env,
        memory=memory,
        use_random_actions=bool(settings.worker_settings.io.actor_weights_path),
        num_transitions=settings.worker_settings.collect_num_transitions,
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
    with AtomicFileWriter(settings.worker_settings.io.result_output_path) as f:
        with open(f, "w") as fw:
            json.dump(result.model_dump(), fw)


def run_evaluation(
    configs: ConfigStack,
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

    Things to override for through CLI from trainer:
    1. model.id
    2. runner.mode
    3. runner.worker.task
    4. runner.worker.actor_weights_path
    5. runner.worker.result_output_path
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
    actor.to(algorithm_config.device)

    # load the weights file and clean up
    if settings.worker_settings.io.actor_weights_path:
        actor.load(settings.worker_settings.io.actor_weights_path)

    # run an eval task
    eval_score, info = evaluation_fn(
        actor=actor,
        env=env,
        num_episodes=settings.worker_settings.eval_num_episodes,
    )

    # form the results
    result = EvaluationResult(
        score=eval_score,
        info=info,
    )

    # dump the pointer to disk
    with AtomicFileWriter(settings.worker_settings.io.result_output_path) as f:
        with open(f, "w") as fw:
            json.dump(result.model_dump(), fw)
