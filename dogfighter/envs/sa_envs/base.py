import dataclasses
from abc import abstractmethod
from typing import Any, ClassVar

from gymnasium import Env
from pydantic import BaseModel, StrictBool, StrictStr


class SAEnvConfig(BaseModel):
    """Initializer for a single agent Gymnasium environment."""

    _registry: ClassVar[set[str]] = set()
    variant: StrictStr
    env_id: StrictStr
    render_mode: StrictStr | StrictBool | None = None
    env_kwargs: dict[StrictStr, Any] = dataclasses.field(default_factory=dict)

    @abstractmethod
    def instantiate(self) -> Env:
        raise NotImplementedError

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        variant = cls.__annotations__.get("variant")
        assert variant is not None
        if variant in cls._registry:
            raise ValueError(f"`variant` {variant} is already in use by another class.")
        cls._registry.add(variant)
