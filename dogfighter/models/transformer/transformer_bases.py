import torch
from pydantic import Field, StrictFloat, StrictInt

from dogfighter.models.bases import (EnvParams, LearningParams, ModelParams,
                                     Observation)


class TransformerEnvParams(EnvParams):
    obs_size: StrictInt
    att_size: StrictInt
    act_size: StrictInt


class TransformerModelParams(ModelParams):
    qu_num_ensemble: StrictInt = Field(2)
    embed_dim: StrictInt = Field(128)
    att_inner_dim: StrictInt = Field(256)
    att_num_heads: StrictInt = Field(2)
    att_num_encoder_layers: StrictInt = Field(2)
    att_num_decoder_layers: StrictInt = Field(2)


class TransformerLearningParams(LearningParams):
    learning_rate: StrictFloat = Field(0.003)
    alpha_learning_rate: StrictFloat = Field(0.01)
    target_entropy: None | StrictFloat = Field(None)
    discount_factor: StrictFloat = Field(0.99)
    update_ratio: StrictInt = Field(1)
    actor_update_ratio: StrictInt = Field(1)
    critic_update_ratio: StrictInt = Field(1)


class TransformerObservation(Observation):
    # other agent observations
    # shape is [B, num_other_agents, obs_size]
    obs: torch.Tensor
    # other agent mask
    # TODO
    obs_mask: torch.Tensor
    # current agent attitude
    # shape is [B, att_size]
    att: torch.Tensor
