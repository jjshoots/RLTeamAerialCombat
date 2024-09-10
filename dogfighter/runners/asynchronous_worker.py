import os
import tempfile
from typing import Any

from wingman import Wingman

from dogfighter.runners.asynchronous_runner import AsynchronousRunnerSettings
from setup_configs import get_all_configs


def run_collection(
    wm: Wingman,
    actor_weight_path: str,
) -> tuple[str, dict[str, Any]]:
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
    env = train_env_config.instantiate()
    actor = algorithm_config.actor_config.instantiate()
    memory = memory_config.instantiate()
    collection_fn = interactor_config.get_collection_fn()

    # get latest weight files if it exists, and clean up
    if actor_weight_path:
        actor.load(actor_weight_path)
        os.remove(actor_weight_path)

    # run a collect task
    memory, info = collection_fn(
        actor=actor.actor,
        env=env,
        memory=memory,
        use_random_actions=actor_weight_path is not None,
        num_transitions=runner_settings.transitions_per_epoch,
    )

    # dump the memory to disk
    fd, memory_path = tempfile.mkstemp(suffix=".zip")
    with open(memory_path, "w+b") as f:
        memory.dump(f)
    os.close(fd)

    return memory_path, info


def run_evaluation(
    wm: Wingman,
    actor_weight_path: str,
) -> tuple[float, dict[str, Any]]:
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
