from dataclasses import field
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, StrictFloat, StrictInt, StrictStr

from dogfighter.env_interactors.base import UpdateInfos

#############################################################
# RESULTS
#############################################################


class CollectionResult(BaseModel):
    memory_path: StrictStr
    info: UpdateInfos


class EvaluationResult(BaseModel):
    score: StrictFloat
    info: UpdateInfos


#############################################################
# SETTINGS
#############################################################


class IOSettings(BaseModel):
    actor_weights_path: StrictStr | None = None
    result_output_path: StrictStr | None = None


class WorkerTaskType(IntEnum):
    NULL = 0
    COLLECT = 1
    EVAL = 2


class WorkerSettings(BaseModel):
    io: IOSettings = field(default_factory=IOSettings)
    task: WorkerTaskType = WorkerTaskType.NULL
    collect_min_transitions: StrictInt
    collect_buffer_size: StrictInt
    eval_num_episodes: StrictInt


class TrainerSettings(BaseModel):
    transitions_max: StrictInt
    transitions_num_exploration: StrictInt
    transitions_min_for_train: StrictInt


class AsynchronousRunnerSettings(BaseModel):
    mode: Literal["trainer", "worker"]
    max_workers: StrictInt
    max_queued_evals: StrictInt

    trainer: TrainerSettings
    worker: WorkerSettings
