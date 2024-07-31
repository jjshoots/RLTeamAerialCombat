from __future__ import annotations

import math

from pydantic import BaseModel
import torch
from wingman import Wingman
from wingman.replay_buffer import ReplayBuffer

from dogfighter.bases.base_algorithm import Algorithm
from dogfighter.bases.base_env_interactors import (CollectFunctionProtocol,
                                                   EvaluationFunctionProtocol,
                                                   SupportedEnvTypes)


class SynchronousRunnerSettings(BaseModel):
    """SynchronousRunnerSettings."""

    max_transitions: int
    transitions_per_epoch: int
    transitions_num_exploration: int

    train_min_transitions: int
    train_steps_per_epoch: int
    train_batch_size: int

    eval_num_episodes: int
    eval_transitions_frequency: int


def run_synchronous(
    wm: Wingman,
    train_env: SupportedEnvTypes,
    eval_env: SupportedEnvTypes,
    collect_fn: CollectFunctionProtocol,
    evaluation_fn: EvaluationFunctionProtocol,
    algorithm: Algorithm,
    memory: ReplayBuffer,
    settings: SynchronousRunnerSettings,
) -> None:
    """A synchronous runner to perform train and evaluations in step.

    Args:
        wm (Wingman): wm
        train_env (SupportedEnvTypes): train_env
        eval_env (SupportedEnvTypes): eval_env
        collect_fn (CollectFunctionProtocol): collect_fn
        evaluation_fn (EvaluationFunctionProtocol): evaluation_fn
        algorithm (Algorithm): algorithm
        memory (ReplayBuffer): memory
        settings (SynchronousRunnerSettings): settings

    Returns:
        None:
    """
    # logging metrics
    num_epochs = 0
    eval_score = -math.inf
    max_eval_score = -math.inf
    next_eval_step = 0

    # start the main training loop
    while memory.count <= settings.max_transitions:
        print("\n\n")
        print(
            f"New epoch @ {memory.count} / {settings.max_transitions} total transitions."
        )
        num_epochs += 1

        """POLICY ROLLOUT"""
        memory, info = collect_fn(
            actor=algorithm.actor,
            env=train_env,
            memory=memory,
            use_random_actions=memory.count < settings.transitions_num_exploration,
            num_transitions=settings.transitions_per_epoch,
        )
        wm.log.update({f"collect/{k}": v for k, v in info})

        # don't proceed with training until we have a minimum number of transitions
        if memory.count < settings.train_min_transitions:
            print(
                "Haven't reached minimum number of transitions "
                f"({memory.count} / {settings.train_min_transitions}) "
                "required before training, continuing with sampling..."
            )
            continue

        """TRAINING RUN"""
        print(
            f"Training epoch {wm.log['runner/epoch']}, "
            f"Replay Buffer Capacity {memory.count} / {memory.mem_size}"
        )
        info = algorithm.update(
            memory=memory,
            batch_size=settings.train_batch_size,
            num_gradient_steps=settings.train_steps_per_epoch,
        )
        wm.log.update({f"train/{k}": v for k, v in info})

        """EVALUATE POLICY"""
        if memory.count >= next_eval_step:
            eval_score, info = evaluation_fn(
                actor=algorithm.actor,
                env=eval_env,
                num_episodes=settings.eval_num_episodes,
            )
            wm.log.update({f"eval/{k}": v for k, v in info})
            max_eval_score = max(max_eval_score, eval_score)
            next_eval_step = (
                int(memory.count / settings.eval_transitions_frequency) + 1
            ) * settings.eval_transitions_frequency

        """WANDB"""
        # collect some statistics
        wm.log["runner/epoch"] = 0
        wm.log["runner/memory_size"] = memory.__len__()
        wm.log["runner/num_transitions"] = memory.count
        wm.log["eval/score"] = eval_score
        wm.log["eval/max_score"] = max_eval_score

        # save weights
        to_update, model_file, _ = wm.checkpoint(loss=-eval_score, step=memory.count)
        if to_update:
            torch.save(algorithm.state_dict(), model_file)
