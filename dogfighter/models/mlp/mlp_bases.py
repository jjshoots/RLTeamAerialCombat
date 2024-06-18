import torch
from pydantic import Field, StrictFloat, StrictInt

from dogfighter.models.bases import (EnvParams, LearningParams, ModelParams,
                                     Observation)


class MlpEnvParams(EnvParams):
    obs_size: StrictInt
    act_size: StrictInt


class MlpModelParams(ModelParams):
    qu_num_ensemble: StrictInt = Field(2)
    embed_dim: StrictInt = Field(128)


class MlpLearningParams(LearningParams):
    learning_rate: StrictFloat = Field(0.003)
    alpha_learning_rate: StrictFloat = Field(0.01)
    target_entropy: None | StrictFloat = Field(None)
    discount_factor: StrictFloat = Field(0.99)
    update_ratio: StrictInt = Field(1)
    actor_update_ratio: StrictInt = Field(1)
    critic_update_ratio: StrictInt = Field(1)


class MlpObservation(torch.Tensor, Observation):
    pass
