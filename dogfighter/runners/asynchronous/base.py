from dataclasses import field
from enum import Enum
from typing import Literal

from pydantic import BaseModel, StrictFloat, StrictInt, StrictStr

#############################################################
# RESULTS
#############################################################


class CollectionResult(BaseModel):
    memory_path: StrictStr
    info: dict[StrictStr, StrictFloat | StrictInt | StrictStr]


class EvaluationResult(BaseModel):
    score: StrictFloat
    info: dict[StrictStr, StrictFloat | StrictInt | StrictStr]


#############################################################
# SETTINGS
#############################################################


class TrainerSettings(BaseModel):
    steps_per_epoch: StrictInt


class IOSettings(BaseModel):
    actor_weights_path: StrictStr = ""
    result_output_path: StrictStr = "/dev/null"


class WorkerSettings(BaseModel):
    task: Literal["collect", "eval", "null"]
    collect_num_transitions: StrictInt
    eval_num_episodes: StrictInt
    io: IOSettings = field(default_factory=IOSettings)


class AsynchronousRunnerSettings(BaseModel):
    mode: Literal["trainer", "worker"]
    max_workers: StrictInt

    transitions_eval_frequency: StrictInt
    transitions_num_exploration: StrictInt
    transitions_min_for_train: StrictInt
    transitions_max: StrictInt

    trainer: TrainerSettings
    worker: WorkerSettings


#############################################################
# TASK DEFINITION
#############################################################


class WorkerTaskType(Enum):
    """WorkerMode."""

    COLLECT = 1
    EVAL = 2
