from typing import Union, cast

import torch

NestedTensor = dict[str, Union[torch.Tensor, "NestedTensor"]]


def zeros_from_memory(x: NestedTensor | torch.Tensor) -> NestedTensor | torch.Tensor:
    """Outputs an identical result as the input but zeroed out."""
    if isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = zeros_from_memory(v)
        return result
    else:
        raise NotImplementedError


def copy_from_memory(
    source: NestedTensor | torch.Tensor,
    target: NestedTensor | torch.Tensor,
) -> None:
    if isinstance(target, torch.Tensor):
        source = cast(torch.Tensor, source)
        target.copy_(source)
        return
    elif isinstance(target, dict):
        source = cast(NestedTensor, source)
        for s, t in zip(source.values(), target.values()):
            copy_from_memory(s, t)
        return
    else:
        raise NotImplementedError
