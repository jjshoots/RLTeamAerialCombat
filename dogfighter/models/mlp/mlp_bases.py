from dataclasses import dataclass

import torch
from pydantic import Field, StrictFloat, StrictInt

from dogfighter.models.bases import (AlgorithmParams, EnvParams, ModelParams,
                                     Observation)


class MlpEnvParams(EnvParams):
    obs_size: StrictInt
    act_size: StrictInt


class MlpModelParams(ModelParams):
    qu_num_ensemble: StrictInt = Field(2)
    embed_dim: StrictInt = Field(128)


class MlpLearningParams(AlgorithmParams):
    learning_rate: StrictFloat = Field(0.003)
    alpha_learning_rate: StrictFloat = Field(0.01)
    target_entropy: None | StrictFloat = Field(None)
    discount_factor: StrictFloat = Field(0.99)
    update_ratio: StrictInt = Field(1)
    actor_update_ratio: StrictInt = Field(1)
    critic_update_ratio: StrictInt = Field(1)


@dataclass
class MlpObservation(Observation):
    # shape is [B, obs_size]
    obs: torch.Tensor
