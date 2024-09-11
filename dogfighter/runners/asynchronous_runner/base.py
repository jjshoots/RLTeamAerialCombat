from typing import Literal

from pydantic import BaseModel, StrictFloat, StrictInt, StrictStr

from dogfighter.runners.base import RunnerSettings


class CollectionResult(BaseModel):
    memory_path: StrictStr
    info: dict[StrictStr, StrictFloat | StrictInt | StrictStr]


class EvaluationResult(BaseModel):
    score: StrictFloat
    info: dict[StrictStr, StrictFloat | StrictInt | StrictStr]


class TaskConfig(BaseModel):
    actor_weight_path: StrictStr | None
    results_path: StrictStr


class WorkerSettings(RunnerSettings):
    task: Literal["collect", "eval"]
    collect_num_transitions: StrictInt
    eval_num_episodes: StrictInt
    task_config_path: StrictStr


class TrainerSettings(RunnerSettings):
    num_parallel_rollouts: StrictInt
    transitions_num_exploration: StrictInt
    transitions_min_for_train: StrictInt
    transitions_max: StrictInt
    eval_transitions_frequency: StrictInt


class AsynchronousRunnerSettings(RunnerSettings):
    """AsynchronousRunnerSettings."""

    mode: Literal["trainer", "worker"]
    trainer: TrainerSettings
    worker: WorkerSettings
