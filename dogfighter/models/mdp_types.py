from typing import Literal, Union

import torch

# observation types
TransformerObservation = dict[
    Literal["src", "tgt", "src_mask", "tgt_mask"], torch.Tensor
]
MlpObservation = torch.Tensor

# combined types
Observation = Union[
    TransformerObservation,
    MlpObservation,
]
Action = torch.Tensor
