import math
import time
from signal import SIGINT, signal

from wingman import Wingman
from wingman.utils import shutdown_handler

from dogfighter.runners.base import ConfigStack
from dogfighter.runners.synchronous.base import SynchronousRunnerSettings

signal(SIGINT, shutdown_handler)


def run_synchronous(
    wm: Wingman,
    configs: ConfigStack,
) -> None:
    """A synchronous runner to perform train and evaluations in step.

    Args:
        wm (Wingman): wm
        configs (TaskConfig): configs

    Returns:
        None:
    """
    train_env_config = configs.train_env_config
    eval_env_config = configs.eval_env_config
    algorithm_config = configs.algorithm_config
    memory_config = configs.memory_config
    interactor_config = configs.interactor_config
    settings = configs.runner_settings
    assert isinstance(settings, SynchronousRunnerSettings)

    # instantiate everything
    train_env = train_env_config.instantiate()
    eval_env = eval_env_config.instantiate()
    algorithm = algorithm_config.instantiate()
    memory = memory_config.instantiate()
    collection_fn = interactor_config.get_collection_fn()
    evaluation_fn = interactor_config.get_evaluation_fn()

    # get latest weight files
    has_weights, _, ckpt_dir = wm.get_weight_files()
    if has_weights:
        algorithm.load(ckpt_dir / "weights.pth")

    # logging metrics
    num_epochs = 0
    eval_score = -math.inf
    max_eval_score = -math.inf
    next_eval_step = 0
    train_start_time = time.time()

    # start the main training loop
    while memory.count <= settings.transitions_max:
        print("\n\n")
        print(
            f"New epoch @ {memory.count} / {settings.transitions_max} total transitions."
        )
        num_epochs += 1
        loop_start_time = time.time()

        """POLICY ROLLOUT"""
        memory, info = collection_fn(
            actor=algorithm.actor,
            env=train_env,
            memory=memory,
            use_random_actions=memory.count < settings.transitions_num_exploration,
            num_transitions=settings.transitions_per_epoch,
        )
        wm.log.update({f"collect/{k}": v for k, v in info.items()})

        # don't proceed with training until we have a minimum number of transitions
        if memory.count < settings.transitions_min_for_train:
            print(
                "Haven't reached minimum number of transitions "
                f"({memory.count} / {settings.transitions_min_for_train}) "
                "required before training, continuing with sampling..."
            )
            continue

        """TRAINING RUN"""
        print(
            f"Training epoch {num_epochs}, "
            f"Replay Buffer Capacity {memory.count} / {memory.mem_size}"
        )
        info = algorithm.update(memory=memory)
        wm.log.update({f"train/{k}": v for k, v in info.items()})

        """EVALUATE POLICY"""
        if memory.count >= next_eval_step:
            eval_score, info = evaluation_fn(
                actor=algorithm.actor,
                env=eval_env,
                num_episodes=settings.eval_num_episodes,
            )
            wm.log.update({f"eval/{k}": v for k, v in info.items()})
            max_eval_score = max(max_eval_score, eval_score)
            next_eval_step = (
                int(memory.count / settings.transitions_eval_frequency) + 1
            ) * settings.transitions_eval_frequency
        wm.log["eval/score"] = eval_score
        wm.log["eval/max_score"] = max_eval_score

        """LOGGING"""
        # collect some statistics
        wm.log["runner/epoch"] = num_epochs
        wm.log["runner/memory_size"] = memory.__len__()
        wm.log["runner/num_transitions"] = memory.count
        wm.log["runner/looptime"] = time.time() - loop_start_time
        wm.log["runner/eta_completion"] = (
            time.time() - train_start_time / memory.count
        ) * settings.transitions_max

        # print things
        print(f"ETA to completion: {wm.log['runner/eta_completion']:.0f} seconds...")

        # save weights
        to_update, _, ckpt_dir = wm.checkpoint(loss=-eval_score, step=memory.count)
        if to_update:
            algorithm.save(ckpt_dir / "weights.pth")
