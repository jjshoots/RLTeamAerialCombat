import json
import os
import tempfile
import time

from dogfighter.algorithms.base import AlgorithmConfig
from dogfighter.env_interactors.base import EnvInteractorConfig
from dogfighter.envs.base import MAEnvConfig, SAVecEnvConfig
from dogfighter.replay_buffers.replay_buffer import ReplayBufferConfig
from dogfighter.runners.asynchronous_runner.base import (
    AsynchronousRunnerSettings, CollectionResult, EvaluationResult)


def run_collection(
    train_env_config: SAVecEnvConfig | MAEnvConfig,
    algorithm_config: AlgorithmConfig,
    memory_config: ReplayBufferConfig,
    interactor_config: EnvInteractorConfig,
    settings: AsynchronousRunnerSettings,
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
    # instantiate things
    env = train_env_config.instantiate()
    actor = algorithm_config.actor_config.instantiate()
    collection_fn = interactor_config.get_collection_fn()
    memory = memory_config.model_copy(
        update={"mem_size": int(settings.worker.collect_num_transitions * 1.2)}
    ).instantiate()

    # load the weights file and clean up
    if settings.worker.actor_weights_path:
        actor.load(settings.worker.actor_weights_path)
        os.remove(settings.worker.actor_weights_path)

    # run a collect task
    memory, info = collection_fn(
        actor=actor,
        env=env,
        memory=memory,
        use_random_actions=bool(settings.worker.actor_weights_path),
        num_transitions=settings.worker.collect_num_transitions,
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

    temp_path = f"{settings.worker.result_output_path}.temp.json"
    with open(temp_path, "w") as f:
        json.dump(result.model_dump(), f)
        time.sleep(1)
    os.rename(temp_path, settings.worker.result_output_path)


def run_evaluation(
    eval_env_config: SAVecEnvConfig | MAEnvConfig,
    algorithm_config: AlgorithmConfig,
    interactor_config: EnvInteractorConfig,
    settings: AsynchronousRunnerSettings,
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
    # instantiate things
    env = eval_env_config.instantiate()
    actor = algorithm_config.actor_config.instantiate()
    evaluation_fn = interactor_config.get_evaluation_fn()

    # load the weights file and clean up
    if settings.worker.actor_weights_path:
        actor.load(settings.worker.actor_weights_path)
        os.remove(settings.worker.actor_weights_path)

    # run an eval task
    eval_score, info = evaluation_fn(
        actor=actor,
        env=env,
        num_episodes=settings.worker.eval_num_episodes,
    )

    # form the results
    result = EvaluationResult(
        score=eval_score,
        info=info,
    )

    # dump the pointer to disk, we do a write, then rename
    # this way, the file can't be read while it's being written
    temp_path = f"{settings.worker.result_output_path}.temp.json"
    with open(temp_path, "w") as f:
        json.dump(result.model_dump(), f)
        time.sleep(1)
    os.rename(temp_path, settings.worker.result_output_path)
