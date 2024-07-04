from typing import Literal

import torch
from pydantic import Field, StrictFloat, StrictInt

from dogfighter.models.bases import AlgorithmParams, EnvParams, ModelParams


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


class TransformerLearningParams(AlgorithmParams):
    learning_rate: StrictFloat = Field(0.003)
    alpha_learning_rate: StrictFloat = Field(0.01)
    target_entropy: None | StrictFloat = Field(None)
    discount_factor: StrictFloat = Field(0.99)
    update_ratio: StrictInt = Field(1)
    actor_update_ratio: StrictInt = Field(1)
    critic_update_ratio: StrictInt = Field(1)


TransformerObservation = dict[Literal["key", "mask", "query"], torch.Tensor]
"""
obs (torch.Tensor): a [B, N, obs_size] tensor for N other UAVs.
obs_mask (torch.Tensor): a [B, N, 1] mask tensor for N other UAVs, where True means the observation is null.
att (torch.Tensor): a [B, att_size] tensor for the current UAV's attitude.
"""
