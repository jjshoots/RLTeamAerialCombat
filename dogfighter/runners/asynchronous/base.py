from enum import IntEnum
from typing import ClassVar, Literal, Union

from pydantic import BaseModel, StrictFloat, StrictInt, StrictStr

from dogfighter.env_interactors.base import UpdateInfos

#############################################################
# TASKS
#############################################################


class TaskDefinition(BaseModel):
    _registry: ClassVar[set[str]] = set()
    variant: StrictStr = "null"

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        variant = cls.__annotations__.get("variant")
        assert variant is not None
        if variant in cls._registry:
            raise ValueError(f"`variant` {variant} is already in use by another class.")
        cls._registry.add(variant)


class CollectionTask(TaskDefinition):
    variant: Literal["collection"] = "collection"  # pyright: ignore

    min_transitions: StrictInt
    buffer_size: StrictInt


class EvaluationTask(TaskDefinition):
    variant: Literal["evaluation"] = "evaluation"  # pyright: ignore

    num_episodes: StrictInt


#############################################################
# RESULTS
#############################################################


class ResultDefinition(BaseModel):
    _registry: ClassVar[set[str]] = set()
    variant: StrictStr = "null"

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        variant = cls.__annotations__.get("variant")
        assert variant is not None
        if variant in cls._registry:
            raise ValueError(f"`variant` {variant} is already in use by another class.")
        cls._registry.add(variant)


class CollectionResult(ResultDefinition):
    variant: Literal["collection"] = "collection"  # pyright: ignore

    memory_path: StrictStr
    info: UpdateInfos


class EvaluationResult(ResultDefinition):
    variant: Literal["evaluation"] = "evaluation"  # pyright: ignore

    score: StrictFloat
    info: UpdateInfos


Result = Union[CollectionResult, EvaluationResult]

#############################################################
# SETTINGS
#############################################################


class TaskType(IntEnum):
    NULL = 0
    COLLECT = 1
    EVAL = 2


class TrainerSettings(BaseModel):
    transitions_max: StrictInt
    transitions_num_exploration: StrictInt
    transitions_min_for_train: StrictInt


class AsynchronousRunnerSettings(BaseModel):
    max_workers: StrictInt
    max_queued_evals: StrictInt

    trainer: TrainerSettings

    # TODO: change this to:
    # task_definitions: list[TaskDefinition]
    # and implement a sampler
    collect: CollectionTask
    evaluate: EvaluationTask
